// Contact atlas: lazy column-grid cache (architect C). The 2 m column grid of atlas_build.h (E), built
// per 16 x 16-column superblock on first use and kept under a byte budget, shared by the threads of a
// batch / search run (one cache per world, read-mostly).
//
//   - Values: a superblock is E's raw layout for those columns (ColumnBlock), a pure function of the
//     world and (bx, bz). Lookups therefore return the same bits whether the block was built earlier,
//     later, by another thread, or rebuilt after an eviction.
//   - Reads are lock-free: one atomic pointer per superblock. A miss takes the block's build lock (one
//     of 64 stripes), re-checks, builds and publishes the block; other readers needing it wait for that
//     build instead of repeating it.
//   - Eviction: least recently used blocks (by tick) beyond the budget are unlinked and retired; a
//     retired block is freed once every active reader has begun a later tick (epoch-based
//     reclamation). A reader's pointers stay valid until its next BeginTick.
//   - Cost accounting: TSC per build, the largest build, and per reader the largest build time inside
//     one tick (the worst first-touch latency).
#ifndef TW_ATLAS_COLCACHE_H
#define TW_ATLAS_COLCACHE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "atlas_build.h"

namespace tw::atlas {

constexpr int kColBlock = kColumnBlock;               // columns per superblock edge (E's)
constexpr int kColBlockSlots = kColBlock * kColBlock * 2;
static_assert(kColBlockSlots <= 512, "a handle keeps the local slot in 9 bits");

// Bytes held by one block (E's ColumnBlock, atlas_build.h).
inline std::size_t BlockBytes(const ColumnBlock &b) {
    return sizeof(b) + b.entries.capacity() * sizeof(ColumnEntry) + b.unions.capacity() * 4 + b.memFirst.capacity() * 4 + b.members.capacity() * 4 +
           b.overlap.capacity();
}

struct ColumnCacheStats {
    std::uint64_t builds = 0, raceDuplicates = 0, evictions = 0, freed = 0;
    std::uint64_t buildTsc = 0, maxBuildTsc = 0, maxTickBuildTsc = 0;   // max per tick: a reader's stalls on misses (build or wait)
    std::uint64_t prefetchBuilds = 0, missStalls = 0;
    std::size_t residentBytes = 0, peakResidentBytes = 0, retiredBytes = 0;
};

class ColumnCache {
public:
    // Builds (bx, bz) into out (E's Atlas::BuildColumnBlock); false outside the grid. Must be pure and thread-safe.
    using Builder = std::function<bool(std::uint32_t bx, std::uint32_t bz, ColumnBlock *out)>;
    struct Options {
        std::size_t budgetBytes = std::size_t(64) << 20;   // resident blocks (0: unbounded)
        int maxReaders = 64;
        bool prefetchThread = false;                       // a helper thread that builds the blocks Prefetch names
    };

    ColumnCache(float x0, float z0, std::uint32_t nx, std::uint32_t nz, Builder build, const Options &opt);
    ~ColumnCache();
    ColumnCache(const ColumnCache &) = delete;
    ColumnCache &operator=(const ColumnCache &) = delete;

    // ---- readers (one per thread) ----
    int Register();                        // a reader id; -1 when all slots are taken
    void BeginTick(int reader);            // pointers obtained before this call become invalid
    void Idle(int reader);                 // the reader holds no pointers (does not hold back reclamation)

    // E's ColumnSlot, as a division-free handle: block index << 9 | local slot (or UINT32_MAX outside)
    std::uint32_t SlotHandle(const float x[3], int cls) const {
        const float fx = (x[0] - x0_) * (1.0f / kColumnM), fz = (x[2] - z0_) * (1.0f / kColumnM);
        if (!(fx >= 0.0f && fz >= 0.0f)) return 0xFFFFFFFFu;
        const std::uint32_t ix = static_cast<std::uint32_t>(fx), iz = static_cast<std::uint32_t>(fz);
        if (ix >= nx_ || iz >= nz_) return 0xFFFFFFFFu;
        return ((iz / kColBlock) * bnx_ + ix / kColBlock) << 9 | (((iz % kColBlock) * kColBlock + ix % kColBlock) * 2u + std::uint32_t(cls));
    }
    // E's slot number ((iz * nx + ix) * 2 + cls) -> handle
    std::uint32_t HandleOf(std::uint32_t slot) const {
        const std::uint32_t col = slot >> 1, ix = col % nx_, iz = col / nx_;
        return ((iz / kColBlock) * bnx_ + ix / kColBlock) << 9 | (((iz % kColBlock) * kColBlock + ix % kColBlock) * 2u + (slot & 1u));
    }

    // The block of a handle, built on first use (null outside the grid).
    const ColumnBlock *Block(int reader, std::uint32_t handle);
    // E's accessors (atlas_build.h Atlas::Entries / Union / EntryMembers), entry = index within the slot.
    const ColumnEntry *Entries(int reader, std::uint32_t handle, std::uint32_t *count);
    const std::int32_t *Union(int reader, std::uint32_t handle, std::uint32_t *count);
    const std::uint32_t *EntryMembers(int reader, std::uint32_t handle, std::uint32_t entry, std::uint32_t *count, bool *overlap);
    // A resolved slot: resolve once per tick per shape (valid until the reader's next BeginTick), then
    // read the slot's entries and union with no further table lookups.
    struct SlotRef {
        const ColumnEntry *entries = nullptr;
        const std::int32_t *unions = nullptr;
        std::uint32_t n = 0, unionCount = 0;
        const ColumnBlock *b = nullptr;
        std::uint32_t first = 0;   // the slot's first entry within the block (members)
    };
    SlotRef Resolve(int reader, std::uint32_t handle) {
        SlotRef s;
        const ColumnBlock *b = Block(reader, handle);
        if (!b) return s;
        const std::uint32_t q = handle & 511u;
        s.b = b;
        s.first = b->first[q];
        s.n = b->first[q + 1] - s.first;
        s.entries = b->entries.data() + s.first;
        s.unionCount = b->unionFirst[q + 1] - b->unionFirst[q];
        s.unions = b->unions.data() + 2u * b->unionFirst[q];
        return s;
    }
    static const std::uint32_t *EntryMembers(const SlotRef &s, std::uint32_t entry, std::uint32_t *count, bool *overlap) {
        if (!s.b) { *count = 0; *overlap = false; return nullptr; }
        const std::uint32_t e = s.first + entry;
        *count = s.b->memFirst[e + 1] - s.b->memFirst[e];
        *overlap = s.b->overlap[e] != 0;
        return s.b->members.data() + s.b->memFirst[e];
    }

    // Ask the helper thread (Options::prefetchThread) to build the block of a handle ahead of use. Pure
    // speculation: lookups give the same bits whether a block came from here or from a miss.
    void Prefetch(std::uint32_t handle);

    ColumnCacheStats Stats() const;

private:
    struct Slot {
        std::atomic<ColumnBlock *> block{nullptr};
        std::atomic<std::uint64_t> lastUse{0};
        std::atomic<bool> queued{false};
    };
    struct Retired {
        ColumnBlock *block;
        std::uint64_t epoch;
    };
    struct alignas(64) Reader {
        std::atomic<std::uint64_t> epoch{~std::uint64_t(0)};   // idle
        std::atomic<bool> used{false};
        std::uint64_t tick = 0;              // LRU clock value of the reader's current tick
        std::uint64_t tickBuildTsc = 0;
    };
    float x0_, z0_;
    std::uint32_t nx_, nz_, bnx_, bnz_;
    Builder build_;
    Options opt_;
    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<Reader[]> readers_;
    std::atomic<std::uint64_t> epoch_{1};   // reclamation epoch (advances at each eviction)
    std::atomic<std::uint64_t> clock_{1};   // LRU clock (advances at each BeginTick)
    std::atomic<std::size_t> resident_{0};
    std::mutex evictMutex_;                 // eviction and the retire list
    std::mutex buildMutex_[64];             // per block (striped): one build of a block at a time
    std::vector<Retired> retired_;
    mutable std::mutex statsMutex_;
    ColumnCacheStats stats_;
    ColumnBlock *Get(int reader, std::uint32_t block);
    ColumnBlock *Build(std::uint32_t block, std::uint64_t now, bool *built);   // under the block's build lock
    // prefetch
    std::mutex queueMutex_;
    std::condition_variable queueCv_;
    std::vector<std::uint32_t> queue_;
    bool stop_ = false;
    std::thread helper_;
    void HelperLoop();
    void Evict(std::uint64_t nowTick);
    void Reclaim();
};

}  // namespace tw::atlas

#endif
