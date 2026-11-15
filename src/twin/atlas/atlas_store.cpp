// Contact atlas tile store (architect C). See atlas_store.h.
#include "atlas_store.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <new>
#include <unordered_map>

namespace tw::atlas {

namespace {

enum PlaneMode : std::uint8_t { kUniform = 0, kPal4 = 1, kPal8 = 2, kRawPairs = 3 };

inline std::uint32_t Pair(std::int16_t lo, std::int16_t hi) { return std::uint32_t(std::uint16_t(lo)) | (std::uint32_t(std::uint16_t(hi)) << 16); }
inline std::int16_t PairLo(std::uint32_t p) { return static_cast<std::int16_t>(p & 0xFFFFu); }
inline std::int16_t PairHi(std::uint32_t p) { return static_cast<std::int16_t>(p >> 16); }

CellRec MakeAlwaysFail() {
    CellRec r;
    for (int k = 0; k < kClassSlots; ++k) { r.lo[k] = 32767; r.hi[k] = -32767; }
    return r;
}
const CellRec kFail = MakeAlwaysFail();

struct Key {
    std::uint32_t plate;
    std::int32_t tx, ty;
    bool operator==(const Key &o) const { return plate == o.plate && tx == o.tx && ty == o.ty; }
};
struct KeyHash {
    std::size_t operator()(const Key &k) const {
        return (std::size_t(k.plate) * 0x9E3779B97F4A7C15ull) ^ (std::size_t(std::uint32_t(k.tx)) << 21) ^ std::size_t(std::uint32_t(k.ty));
    }
};

}  // namespace

// One tile: a 128-byte header, then the plane data, then the member lists, in one allocation.
struct alignas(64) EncodedTile {
    CellRec uniformRec;               // uniform tile: the record of every cell
    std::uint8_t uniform;             // 1: every cell is uniformRec (planes unused)
    std::uint8_t mode[kClassSlots];   // per class slot: PlaneMode
    std::uint8_t wideFirst;           // member offsets are u32 (else u16)
    std::uint8_t pad0[2];
    std::uint32_t planeOff[kClassSlots];   // byte offset of the plane from the tile base
    std::uint32_t pair[kClassSlots];       // kUniform planes: the pair
    std::uint32_t firstOff, membersOff;    // member lists (firstOff == 0: no members)
    std::uint32_t bytes;                   // whole allocation
    std::uint64_t lastUse;                 // LRU clock (lazy stores)

    const std::uint8_t *Base() const { return reinterpret_cast<const std::uint8_t *>(this); }
    inline std::uint32_t PairAt(int cls, int cell) const {
        const std::uint8_t *p = Base() + planeOff[cls];
        switch (mode[cls]) {
            case kUniform: return pair[cls];
            case kPal4: return reinterpret_cast<const std::uint32_t *>(p + 512)[(p[cell >> 1] >> ((cell & 1) * 4)) & 15u];
            case kPal8: return reinterpret_cast<const std::uint32_t *>(p + 1024)[p[cell]];
            default: return reinterpret_cast<const std::uint32_t *>(p)[cell];
        }
    }
};
static_assert(sizeof(EncodedTile) == 128, "EncodedTile header: 128 bytes");

struct Store::Impl {
    std::unordered_map<Key, EncodedTile *, KeyHash> tiles;
    Builder build;
    std::size_t budget = 0;
    std::unordered_map<Key, bool, KeyHash> missing;   // lazy: keys the builder reported absent
    thread_local static CellRec decode;
    ~Impl() {
        for (auto &kv : tiles) Free(kv.second);
    }
    static void Free(EncodedTile *t) {
        t->~EncodedTile();
        ::operator delete(static_cast<void *>(t), std::align_val_t(64));
    }
};
thread_local CellRec Store::Impl::decode;

namespace {

// Encodes one tile (see the header): returns the allocation.
EncodedTile *Encode(const RawTile &raw, StoreStats *st) {
    // uniform?
    bool uni = true;
    for (int c = 1; c < kCellsPerTile && uni; ++c) uni = std::memcmp(&raw.cell[c], &raw.cell[0], sizeof(CellRec)) == 0;
    // planes: per class the pair plane, its mode and size; equal planes share
    std::vector<std::uint32_t> plane[kClassSlots];
    std::uint8_t mode[kClassSlots] = {};
    std::uint32_t pairU[kClassSlots] = {};
    int sameAs[kClassSlots];
    std::vector<std::uint32_t> pal[kClassSlots];
    std::size_t size[kClassSlots] = {};
    for (int k = 0; k < kClassSlots; ++k) {
        sameAs[k] = -1;
        if (uni) continue;
        plane[k].resize(kCellsPerTile);
        for (int c = 0; c < kCellsPerTile; ++c) plane[k][static_cast<std::size_t>(c)] = Pair(raw.cell[c].lo[k], raw.cell[c].hi[k]);
        for (int j = 0; j < k; ++j)
            if (plane[j] == plane[k]) { sameAs[k] = j; break; }
        if (sameAs[k] >= 0) continue;
        std::vector<std::uint32_t> &p = pal[k];
        p = plane[k];
        std::sort(p.begin(), p.end());
        p.erase(std::unique(p.begin(), p.end()), p.end());
        if (p.size() == 1) { mode[k] = kUniform; pairU[k] = p[0]; size[k] = 0; }
        else if (p.size() <= 16) { mode[k] = kPal4; size[k] = 512 + 64; }
        else if (p.size() <= 256) { mode[k] = kPal8; size[k] = 1024 + 4 * p.size(); }
        else { mode[k] = kRawPairs; size[k] = 4 * kCellsPerTile; }
    }
    // members
    const bool hasMembers = !raw.first.empty() && !raw.members.empty();
    const bool wide = hasMembers && raw.members.size() > 65535;
    const std::size_t firstBytes = hasMembers ? (wide ? 4 : 2) * (kCellsPerTile + 1) : 0;
    const std::size_t memberBytes = hasMembers ? 4 * raw.members.size() : 0;
    // layout
    std::size_t off = sizeof(EncodedTile);
    std::uint32_t planeOff[kClassSlots] = {};
    for (int k = 0; k < kClassSlots; ++k) {
        if (uni || sameAs[k] >= 0 || mode[k] == kUniform) continue;
        off = (off + 63) & ~std::size_t(63);
        planeOff[k] = static_cast<std::uint32_t>(off);
        off += size[k];
    }
    off = (off + 3) & ~std::size_t(3);
    const std::size_t firstOff = hasMembers ? off : 0;
    off += firstBytes;
    off = (off + 3) & ~std::size_t(3);
    const std::size_t membersOff = hasMembers ? off : 0;
    off += memberBytes;
    const std::size_t bytes = (off + 63) & ~std::size_t(63);
    void *mem = ::operator new(bytes, std::align_val_t(64));
    std::memset(mem, 0, bytes);
    EncodedTile *t = new (mem) EncodedTile;
    std::uint8_t *base = static_cast<std::uint8_t *>(mem);
    t->uniform = uni ? 1 : 0;
    t->uniformRec = raw.cell[0];
    t->bytes = static_cast<std::uint32_t>(bytes);
    t->lastUse = 0;
    for (int k = 0; k < kClassSlots; ++k) {
        if (uni) { t->mode[k] = kUniform; t->pair[k] = Pair(raw.cell[0].lo[k], raw.cell[0].hi[k]); continue; }
        const int s = sameAs[k] >= 0 ? sameAs[k] : k;
        t->mode[k] = mode[s];
        t->pair[k] = pairU[s];
        t->planeOff[k] = planeOff[s];
        if (st) {
            if (sameAs[k] >= 0) ++st->sharedPlanes;
            ++st->planes[mode[s]];
        }
        if (sameAs[k] >= 0 || mode[k] == kUniform) continue;
        std::uint8_t *p = base + planeOff[k];
        const std::vector<std::uint32_t> &pl = pal[k];
        auto indexOf = [&](std::uint32_t v) { return static_cast<std::uint32_t>(std::lower_bound(pl.begin(), pl.end(), v) - pl.begin()); };
        if (mode[k] == kPal4) {
            for (int c = 0; c < kCellsPerTile; ++c) p[c >> 1] |= static_cast<std::uint8_t>(indexOf(plane[k][static_cast<std::size_t>(c)]) << ((c & 1) * 4));
            std::memcpy(p + 512, pl.data(), 4 * pl.size());
        } else if (mode[k] == kPal8) {
            for (int c = 0; c < kCellsPerTile; ++c) p[c] = static_cast<std::uint8_t>(indexOf(plane[k][static_cast<std::size_t>(c)]));
            std::memcpy(p + 1024, pl.data(), 4 * pl.size());
        } else {
            std::memcpy(p, plane[k].data(), 4 * kCellsPerTile);
        }
    }
    t->wideFirst = wide ? 1 : 0;
    t->firstOff = static_cast<std::uint32_t>(firstOff);
    t->membersOff = static_cast<std::uint32_t>(membersOff);
    if (hasMembers) {
        if (wide) std::memcpy(base + firstOff, raw.first.data(), 4 * (kCellsPerTile + 1));
        else
            for (int c = 0; c <= kCellsPerTile; ++c) reinterpret_cast<std::uint16_t *>(base + firstOff)[c] = static_cast<std::uint16_t>(raw.first[static_cast<std::size_t>(c)]);
        std::memcpy(base + membersOff, raw.members.data(), memberBytes);
    }
    if (st) {
        ++st->tiles;
        st->uniformTiles += uni;
        st->rawBytes += sizeof(CellRec) * kCellsPerTile;
        st->encodedBytes += bytes;
        st->memberRawBytes += hasMembers ? 4 * (kCellsPerTile + 1) + memberBytes : 0;
        st->memberBytes += firstBytes + memberBytes;
    }
    return t;
}

}  // namespace

const CellRec &AlwaysFail() { return kFail; }

Store::Store() : impl_(new Impl) {}
Store::~Store() = default;

void Store::Reset(std::uint32_t plateCount) {
    (void)plateCount;
    impl_.reset(new Impl);
    stats_ = StoreStats{};
    clock_ = 0;
}

void Store::Add(std::uint32_t plate, std::int32_t tx, std::int32_t ty, const RawTile &tile) {
    const Key key{plate, tx, ty};
    auto it = impl_->tiles.find(key);
    if (it != impl_->tiles.end()) Impl::Free(it->second);
    impl_->tiles[key] = Encode(tile, &stats_);
    stats_.residentBytes = stats_.encodedBytes;
    stats_.peakResidentBytes = std::max(stats_.peakResidentBytes, stats_.residentBytes);
}

void Store::Seal() { stats_.indexBytes = impl_->tiles.size() * (sizeof(Key) + sizeof(void *)) * 2; }

void Store::SetBuilder(Builder build, std::size_t budgetBytes) {
    impl_->build = std::move(build);
    impl_->budget = budgetBytes;
}

TileRef Store::Tile(std::uint32_t plate, std::int32_t tx, std::int32_t ty) {
    const Key key{plate, tx, ty};
    auto it = impl_->tiles.find(key);
    if (it != impl_->tiles.end()) {
        it->second->lastUse = clock_;
        return TileRef{it->second};
    }
    if (!impl_->build) return TileRef{};
    if (impl_->missing.count(key)) return TileRef{};
    thread_local RawTile raw;
    raw.first.clear();
    raw.members.clear();
    if (!impl_->build(plate, tx, ty, &raw)) {
        impl_->missing.emplace(key, true);
        return TileRef{};
    }
    EncodedTile *t = Encode(raw, &stats_);
    t->lastUse = clock_;
    ++stats_.materialised;
    stats_.residentBytes += t->bytes;
    if (impl_->budget && stats_.residentBytes > impl_->budget) {   // evict the least recently used tiles (not this tick's)
        std::vector<std::pair<std::uint64_t, Key>> order;
        order.reserve(impl_->tiles.size());
        for (auto &kv : impl_->tiles) order.push_back({kv.second->lastUse, kv.first});
        std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (const auto &o : order) {
            if (stats_.residentBytes <= impl_->budget * 3 / 4 || o.first >= clock_) break;
            auto e = impl_->tiles.find(o.second);
            stats_.residentBytes -= e->second->bytes;
            Impl::Free(e->second);
            impl_->tiles.erase(e);
            ++stats_.evicted;
        }
    }
    impl_->tiles.emplace(key, t);
    stats_.peakResidentBytes = std::max(stats_.peakResidentBytes, stats_.residentBytes);
    return TileRef{t};
}

const CellRec *Store::Cell(TileRef t, int cell) const {
    if (!t.t) return &kFail;
    if (t.t->uniform) return &t.t->uniformRec;
    CellRec &r = Impl::decode;
    for (int k = 0; k < kClassSlots; ++k) {
        const std::uint32_t p = t.t->PairAt(k, cell);
        r.lo[k] = PairLo(p);
        r.hi[k] = PairHi(p);
    }
    return &r;
}

void Store::GatherClass(const TileRef *t, const int *cell, const int *cls, int n, std::int16_t *lo, std::int16_t *hi) const {
    for (int i = 0; i < n; ++i) {
        const EncodedTile *e = t[i].t;
        if (!e) { lo[i] = 32767; hi[i] = -32767; continue; }
        if (e->uniform) { lo[i] = e->uniformRec.lo[cls[i]]; hi[i] = e->uniformRec.hi[cls[i]]; continue; }
        const std::uint32_t p = e->PairAt(cls[i], cell[i]);
        lo[i] = PairLo(p);
        hi[i] = PairHi(p);
    }
}

MemberSpan Store::Members(TileRef t, int cell) const {
    if (!t.t || t.t->firstOff == 0) return MemberSpan{};
    const std::uint8_t *b = t.t->Base();
    std::uint32_t f0, f1;
    if (t.t->wideFirst) {
        const std::uint32_t *f = reinterpret_cast<const std::uint32_t *>(b + t.t->firstOff);
        f0 = f[cell]; f1 = f[cell + 1];
    } else {
        const std::uint16_t *f = reinterpret_cast<const std::uint16_t *>(b + t.t->firstOff);
        f0 = f[cell]; f1 = f[cell + 1];
    }
    return MemberSpan{reinterpret_cast<const std::uint32_t *>(b + t.t->membersOff) + f0, f1 - f0};
}

bool Store::Equal(TileRef t, const RawTile &raw) const {
    for (int c = 0; c < kCellsPerTile; ++c) {
        const CellRec *r = Cell(t, c);
        if (std::memcmp(r, &raw.cell[c], sizeof(CellRec)) != 0) return false;
        for (int k = 0; k < kClassSlots; ++k) {
            const int cl[1] = {k}, ce[1] = {c};
            std::int16_t lo, hi;
            GatherClass(&t, ce, cl, 1, &lo, &hi);
            if (lo != raw.cell[c].lo[k] || hi != raw.cell[c].hi[k]) return false;
        }
        const MemberSpan m = Members(t, c);
        const bool hasRaw = !raw.first.empty() && !raw.members.empty();
        const std::uint32_t n = hasRaw ? raw.first[static_cast<std::size_t>(c) + 1] - raw.first[static_cast<std::size_t>(c)] : 0;
        if (m.size != n) return false;
        if (n && std::memcmp(m.data, raw.members.data() + raw.first[static_cast<std::size_t>(c)], 4 * n) != 0) return false;
    }
    return true;
}

}  // namespace tw::atlas
