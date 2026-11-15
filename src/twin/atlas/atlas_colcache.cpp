// Contact atlas: lazy column-grid cache (architect C). See atlas_colcache.h.
#include "atlas_colcache.h"

#include <algorithm>
#include <x86intrin.h>

namespace tw::atlas {

ColumnCache::ColumnCache(float x0, float z0, std::uint32_t nx, std::uint32_t nz, Builder build, const Options &opt)
    : x0_(x0), z0_(z0), nx_(nx), nz_(nz), bnx_((nx + kColBlock - 1) / kColBlock), bnz_((nz + kColBlock - 1) / kColBlock), build_(std::move(build)), opt_(opt) {
    slots_.reset(new Slot[std::size_t(bnx_) * bnz_]);
    readers_.reset(new Reader[static_cast<std::size_t>(std::max(1, opt_.maxReaders))]);
    if (opt_.prefetchThread) helper_ = std::thread([this] { HelperLoop(); });
}

ColumnCache::~ColumnCache() {
    if (helper_.joinable()) {
        {
            std::lock_guard<std::mutex> g(queueMutex_);
            stop_ = true;
        }
        queueCv_.notify_all();
        helper_.join();
    }
    for (std::size_t i = 0; i < std::size_t(bnx_) * bnz_; ++i) delete slots_[i].block.load();
    for (const Retired &r : retired_) delete r.block;
}

int ColumnCache::Register() {
    for (int i = 0; i < opt_.maxReaders; ++i) {
        bool f = false;
        if (readers_[i].used.compare_exchange_strong(f, true)) return i;
    }
    return -1;
}

void ColumnCache::BeginTick(int reader) {
    Reader &r = readers_[reader];
    {   // the worst build time inside the reader's last tick
        std::lock_guard<std::mutex> g(statsMutex_);
        stats_.maxTickBuildTsc = std::max(stats_.maxTickBuildTsc, r.tickBuildTsc);
    }
    r.tickBuildTsc = 0;
    r.tick = clock_.fetch_add(1, std::memory_order_relaxed);
    r.epoch.store(epoch_.load(std::memory_order_seq_cst), std::memory_order_seq_cst);
}

void ColumnCache::Idle(int reader) { readers_[reader].epoch.store(~std::uint64_t(0), std::memory_order_seq_cst); }

ColumnBlock *ColumnCache::Get(int reader, std::uint32_t block) {
    Slot &s = slots_[block];
    const std::uint64_t now = readers_[reader].tick;
    ColumnBlock *b = s.block.load(std::memory_order_acquire);
    if (b) {
        if (s.lastUse.load(std::memory_order_relaxed) < now) s.lastUse.store(now, std::memory_order_relaxed);
        return b;
    }
    // miss: the reader stalls until the block is there (its own build, or another's it waits for)
    const std::uint64_t t0 = __rdtsc();
    bool built = false;
    b = Build(block, now, &built);
    readers_[reader].tickBuildTsc += __rdtsc() - t0;
    {
        std::lock_guard<std::mutex> g(statsMutex_);
        ++stats_.missStalls;
        if (!built) ++stats_.raceDuplicates;   // builds avoided by waiting
    }
    return b;
}

// One build of a block at a time (striped lock); a caller that waited finds the block published.
ColumnBlock *ColumnCache::Build(std::uint32_t block, std::uint64_t now, bool *built) {
    Slot &s = slots_[block];
    std::lock_guard<std::mutex> bl(buildMutex_[block & 63u]);
    ColumnBlock *b = s.block.load(std::memory_order_acquire);
    if (b) return b;
    const std::uint64_t t0 = __rdtsc();
    std::unique_ptr<ColumnBlock> nb(new ColumnBlock);
    if (!build_(block % bnx_, block / bnx_, nb.get())) return nullptr;
    const std::uint64_t dt = __rdtsc() - t0;
    const std::size_t bytes = BlockBytes(*nb);
    ColumnBlock *mine = nb.release();
    s.lastUse.store(now, std::memory_order_relaxed);
    s.block.store(mine, std::memory_order_release);
    *built = true;
    const std::size_t res = resident_.fetch_add(bytes) + bytes;
    {
        std::lock_guard<std::mutex> g(statsMutex_);
        ++stats_.builds;
        stats_.buildTsc += dt;
        stats_.maxBuildTsc = std::max(stats_.maxBuildTsc, dt);
        stats_.peakResidentBytes = std::max(stats_.peakResidentBytes, res);
    }
    if (opt_.budgetBytes && res > opt_.budgetBytes) Evict(now);
    return mine;
}

void ColumnCache::Prefetch(std::uint32_t handle) {
    if (!opt_.prefetchThread || handle == 0xFFFFFFFFu) return;
    const std::uint32_t block = handle >> 9;
    Slot &s = slots_[block];
    if (s.block.load(std::memory_order_acquire) || s.queued.exchange(true)) return;
    {
        std::lock_guard<std::mutex> g(queueMutex_);
        queue_.push_back(block);
    }
    queueCv_.notify_one();
}

void ColumnCache::HelperLoop() {
    for (;;) {
        std::uint32_t block;
        {
            std::unique_lock<std::mutex> g(queueMutex_);
            queueCv_.wait(g, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            block = queue_.front();
            queue_.erase(queue_.begin());
        }
        bool built = false;
        (void)Build(block, clock_.load(std::memory_order_relaxed), &built);
        slots_[block].queued.store(false);
        if (built) {
            std::lock_guard<std::mutex> g(statsMutex_);
            ++stats_.prefetchBuilds;
        }
    }
}

// Unlinks the least recently used blocks (never one used in the evicting reader's current tick) down to
// 3/4 of the budget, retires them at the current epoch and advances it; then frees what no reader can
// hold. A block another reader is using may be unlinked (it is rebuilt on its next miss) but is freed
// only after that reader's next BeginTick.
void ColumnCache::Evict(std::uint64_t now) {
    std::lock_guard<std::mutex> g(evictMutex_);
    if (resident_.load() <= opt_.budgetBytes) return;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> order;
    const std::uint32_t nb = bnx_ * bnz_;
    for (std::uint32_t i = 0; i < nb; ++i)
        if (slots_[i].block.load(std::memory_order_relaxed)) order.push_back({slots_[i].lastUse.load(std::memory_order_relaxed), i});
    std::sort(order.begin(), order.end());
    const std::uint64_t retireEpoch = epoch_.load();
    std::size_t evicted = 0;
    for (const auto &o : order) {
        if (resident_.load() <= opt_.budgetBytes / 4 * 3) break;
        if (o.first >= now) break;   // used in this tick window: keep
        ColumnBlock *b = slots_[o.second].block.exchange(nullptr, std::memory_order_acq_rel);
        if (!b) continue;
        resident_.fetch_sub(BlockBytes(*b));
        retired_.push_back(Retired{b, retireEpoch});
        ++evicted;
    }
    epoch_.fetch_add(1);   // readers beginning a tick from now on cannot hold the unlinked blocks
    {
        std::lock_guard<std::mutex> s(statsMutex_);
        stats_.evictions += evicted;
    }
    Reclaim();
}

void ColumnCache::Reclaim() {   // evictMutex_ held
    std::uint64_t minEpoch = ~std::uint64_t(0);
    for (int i = 0; i < opt_.maxReaders; ++i)
        if (readers_[i].used.load()) minEpoch = std::min(minEpoch, readers_[i].epoch.load(std::memory_order_seq_cst));
    std::size_t freed = 0, kept = 0;
    std::vector<Retired> keep;
    for (const Retired &r : retired_) {
        if (r.epoch < minEpoch) { delete r.block; ++freed; }
        else { keep.push_back(r); kept += BlockBytes(*r.block); }
    }
    retired_.swap(keep);
    std::lock_guard<std::mutex> s(statsMutex_);
    stats_.freed += freed;
    stats_.retiredBytes = kept;
}

const ColumnBlock *ColumnCache::Block(int reader, std::uint32_t handle) {
    if (handle == 0xFFFFFFFFu) return nullptr;
    return Get(reader, handle >> 9);
}

const ColumnEntry *ColumnCache::Entries(int reader, std::uint32_t handle, std::uint32_t *count) {
    const ColumnBlock *b = Block(reader, handle);
    if (!b) { *count = 0; return nullptr; }
    const std::uint32_t q = handle & 511u;
    *count = b->first[q + 1] - b->first[q];
    return b->entries.data() + b->first[q];
}

const std::int32_t *ColumnCache::Union(int reader, std::uint32_t handle, std::uint32_t *count) {
    const ColumnBlock *b = Block(reader, handle);
    if (!b) { *count = 0; return nullptr; }
    const std::uint32_t q = handle & 511u;
    *count = b->unionFirst[q + 1] - b->unionFirst[q];
    return b->unions.data() + 2u * b->unionFirst[q];
}

const std::uint32_t *ColumnCache::EntryMembers(int reader, std::uint32_t handle, std::uint32_t entry, std::uint32_t *count, bool *overlap) {
    const ColumnBlock *b = Block(reader, handle);
    if (!b) { *count = 0; *overlap = false; return nullptr; }
    const std::uint32_t e = b->first[handle & 511u] + entry;
    *count = b->memFirst[e + 1] - b->memFirst[e];
    *overlap = b->overlap[e] != 0;
    return b->members.data() + b->memFirst[e];
}

ColumnCacheStats ColumnCache::Stats() const {
    std::lock_guard<std::mutex> g(statsMutex_);
    ColumnCacheStats s = stats_;
    s.residentBytes = resident_.load();
    return s;
}

}  // namespace tw::atlas
