// Contact atlas: build from the world package (E, S1). FV-free.
//
//   plates   plane classes of the solid, non-trigger, non-duplicate world triangles (atlas_format.h)
//   members  each plate member in its plate's frame (edge lines, own plane)
//   tiles    per plate, 8 m tiles over the members' footprints: CellRec gaps + per-cell member lists.
//            Built lazily on first use (thread-safe) or all at Build (BuildOptions::allTiles). A tile is a
//            pure function of the world, so lazy and eager builds give the same values.
//   columns  a dense 2 m x-z column grid: per column and radius class, the plates (and loose triangles:
//            triggers, duplicates, degenerate ones) within that radius of the column, with their exact y-range
//            there. A shape that overlaps no entry's y-range is free; the others are its candidate home plates.
//
// The tile and grid storage here is the raw, uncompressed form; architect C's store (atlas_store.*) serves
// the same values from a compressed layout.
#ifndef TW_ATLAS_BUILD_H
#define TW_ATLAS_BUILD_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "atlas_format.h"
#include "tw_accel.h"
#include "tw_world_pack.h"

namespace tw::atlas {

struct BuildOptions {
    bool allTiles = false;     // build every tile at Build (whole-map memory; C's measurement), else lazily
    bool lazyColumns = false;  // skip the whole-grid column build (the frame is still set): C's ColumnCache builds blocks
};

struct Tile {
    CellRec cell[kTileCells * kTileCells];
    std::vector<std::uint32_t> memFirst;   // kTileCells^2 + 1 offsets into memList
    std::vector<std::uint32_t> memList;    // Atlas::members indices whose footprint overlaps the cell (1 mm margin)
    std::vector<std::uint8_t> overlap;     // per cell: 1 when two of its members overlap in the plane (FV may emit two FACEs)
};

// columns: a dense grid of kColumnM x kColumnM columns over the world's x-z bounds (y up). Per column and per
// radius class (wheel: kColumnRadius[0], body: kColumnRadius[1]), the entries of every plate (or of the loose
// triangles: kLoose) whose geometry meets the column's x-z rectangle dilated by the class radius, with that
// geometry's exact y-range there (clipped). A shape of reach <= the class radius, centred in the column, can
// touch only entries whose y-range overlaps [t.y - rmax, t.y + rmax]: none means free.
constexpr float kColumnM = 2.0f;
constexpr float kColumnRadius[2] = {0.40f, 1.95f};
constexpr std::uint32_t kLoose = 0xFFFFFFFEu;
constexpr int kColumnBlock = 16;           // superblock: 16 x 16 columns (C's lazy column cache)
// Canonical values (C's packed store reproduces them bit for bit): y in integer mm, rounded outward (y0 = floor
// - 1 mm, y1 = ceil + 1 mm); the x-z box of the clipped geometry in u8 steps of (kColumnM + 2 R) / 255 from the
// dilated column's min corner, rounded outward. A shape tests [floor(lo mm), ceil(hi mm)] against y and its
// centre +- rmax against the box: both conservative.
struct ColumnEntry {                       // 20 B; per (column, class) sorted by y0
    std::uint32_t plate;                   // plate, or kLoose
    std::int32_t y0, y1;                   // y-range (mm) of that geometry in the dilated column
    std::int32_t y1max;                    // max of y1 over this and the earlier entries (prefix max): the scan starts at
                                           // the first entry with y1max >= the shape's lowest point and stops at y0 > its top
    std::uint8_t box[4];                   // x0, x1, z0, z1 (u8 steps)
};
static_assert(sizeof(ColumnEntry) == 20, "ColumnEntry layout");
inline float ColumnBoxStep(int cls) { return (kColumnM + 2.0f * kColumnRadius[cls]) / 255.0f; }

// One 16 x 16-column superblock in the run-time layout, local indices (C's lazy column cache builds them on demand).
// Local slot q = ((iz % 16) * 16 + ix % 16) * 2 + cls.
struct ColumnBlock {
    std::uint32_t bx = 0, bz = 0;
    std::uint32_t first[2 * kColumnBlock * kColumnBlock + 1];       // local slot -> first entry
    std::uint32_t unionFirst[2 * kColumnBlock * kColumnBlock + 1];  // local slot -> first union interval
    std::vector<ColumnEntry> entries;        // as BuildColumns: per slot sorted by y0, y1max, box
    std::vector<std::int32_t> unions;        // y0, y1 pairs (mm)
    std::vector<std::uint32_t> memFirst;     // per entry + 1
    std::vector<std::uint32_t> members;      // Atlas::Members() indices
    std::vector<std::uint8_t> overlap;       // per entry
};

class Atlas {
public:
    // Pure: the same bits as BuildColumns for those columns, whatever the call order, thread or repetition.
    // Callable concurrently from several threads (read-only on the atlas). False outside the grid.
    bool BuildColumnBlock(std::uint32_t bx, std::uint32_t bz, ColumnBlock *out) const;
    // grid geometry for the cache: x0, z0 (world, the column grid's min corner), nx, nz (columns; blocks follow)
    void ColumnGrid(float *x0, float *z0, std::uint32_t *nx, std::uint32_t *nz) const { *x0 = colX0_; *z0 = colZ0_; *nx = colNx_; *nz = colNz_; }
    bool HasColumnGrid() const { return !colFirst_.empty(); }
    // bvh: a BVH4 over view.tris (the exact stage's); must outlive the atlas.
    bool Build(const world::WorldPackView &view, const Bvh4 &bvh, const BuildOptions &opt, std::string *error);

    const world::WorldPackView &View() const { return *view_; }
    const std::vector<PlateRec> &Plates() const { return plates_; }
    const std::vector<MemberRec> &Members() const { return members_; }
    std::uint32_t PlateOf(std::uint32_t tri) const { return plateOf_[tri]; }

    // the tile (plate, tx, ty), built on first use; null when the plate has no member footprint there
    const Tile *TileAt(std::uint32_t plate, std::int32_t tx, std::int32_t ty) const;
    // the (column, class) slot of a world point, or UINT32_MAX outside the grid
    std::uint32_t ColumnSlot(const float x[3], int cls) const {
        const float fx = (x[0] - colX0_) * (1.0f / kColumnM), fz = (x[2] - colZ0_) * (1.0f / kColumnM);
        if (!(fx >= 0.0f && fz >= 0.0f)) return 0xFFFFFFFFu;
        const std::uint32_t ix = static_cast<std::uint32_t>(fx), iz = static_cast<std::uint32_t>(fz);
        if (ix >= colNx_ || iz >= colNz_) return 0xFFFFFFFFu;
        return (iz * colNx_ + ix) * 2u + static_cast<std::uint32_t>(cls);
    }
    // the slot's entries (sorted by y0) and, separately, the union of all their y-ranges as disjoint
    // intervals (y0, y1 pairs, ascending): the free test
    const ColumnEntry *Entries(std::uint32_t slot, std::uint32_t *count) const {
        *count = colFirst_[slot + 1] - colFirst_[slot];
        return colEntries_.data() + colFirst_[slot];
    }
    const std::int32_t *Union(std::uint32_t slot, std::uint32_t *count) const {
        *count = colUnionFirst_[slot + 1] - colUnionFirst_[slot];
        return colUnion_.data() + 2u * colUnionFirst_[slot];
    }
    // the dilated column's min corner (x, z) of a slot
    void SlotCorner(std::uint32_t slot, float *x0, float *z0) const {
        const std::uint32_t col = slot >> 1, cls = slot & 1u;
        const std::uint32_t ix = col % colNx_, iz = col / colNx_;
        *x0 = colX0_ + float(ix) * kColumnM - kColumnRadius[cls];
        *z0 = colZ0_ + float(iz) * kColumnM - kColumnRadius[cls];
    }
    // wheel-class entries: the plate members whose footprint meets the dilated column (sorted), and whether two
    // of them overlap in the plane
    const std::uint32_t *EntryMembers(std::uint32_t entry, std::uint32_t *count, bool *overlap) const {
        *count = entMemFirst_[entry + 1] - entMemFirst_[entry];
        *overlap = entOverlap_[entry] != 0;
        return entMembers_.data() + entMemFirst_[entry];
    }
    std::uint32_t EntryIndex(const ColumnEntry *e) const { return static_cast<std::uint32_t>(e - colEntries_.data()); }

    struct Stats {
        std::size_t tris = 0, plates = 0, members = 0, loose = 0, dups = 0;
        std::size_t columns = 0, columnEntries = 0, columnUnion = 0, entryMembers = 0;
        double buildMs = 0, gridMs = 0;
    };
    Stats stats;
    std::size_t TilesBuilt() const;
    std::size_t Bytes() const;   // plates + members + grid + tiles built so far

private:
    struct TileKey {
        std::uint32_t plate; std::int32_t tx, ty;
        bool operator==(const TileKey &o) const { return plate == o.plate && tx == o.tx && ty == o.ty; }
    };
    struct TileKeyHash {
        std::size_t operator()(const TileKey &k) const {
            std::uint64_t h = k.plate * 0x9E3779B97F4A7C15ull;
            h ^= (static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.tx)) << 32) ^ static_cast<std::uint32_t>(k.ty);
            h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
            return static_cast<std::size_t>(h);
        }
    };
    const world::WorldPackView *view_ = nullptr;
    const Bvh4 *bvh_ = nullptr;
    std::vector<PlateRec> plates_;
    std::vector<MemberRec> members_;
    std::vector<std::uint32_t> plateOf_;      // per tri: plate, or kNoPlate (loose)
    std::vector<std::uint32_t> memberOf_;     // per tri: members_ index, or kNoPlate
    std::vector<double> triN_;                // per tri: double unit normal (3) and offset (1)
    std::vector<std::uint8_t> loose_;         // per tri: 1 trigger / duplicate / degenerate
    // tiles (lazy)
    mutable std::mutex tileMutex_;
    mutable std::unordered_map<TileKey, std::unique_ptr<Tile>, TileKeyHash> tiles_;
    std::unique_ptr<Tile> BuildTile(std::uint32_t plate, std::int32_t tx, std::int32_t ty) const;
    bool TileHasFootprint(std::uint32_t plate, std::int32_t tx, std::int32_t ty) const;
    // per plate: the tile footprint (tiles within 2.5 m of a member's 2D box), sorted keys tx << 32 | ty
    std::vector<std::vector<std::uint64_t>> footprint_;
    // columns
    float colX0_ = 0.0f, colZ0_ = 0.0f;
    std::uint32_t colNx_ = 0, colNz_ = 0;
    std::vector<std::uint32_t> colFirst_;      // (column * 2 + class) -> first entry; size 2 * columns + 1
    std::vector<ColumnEntry> colEntries_;
    std::vector<std::uint32_t> colUnionFirst_; // slot -> first union interval
    std::vector<std::int32_t> colUnion_;       // y0, y1 pairs (mm)
    std::vector<std::uint32_t> entMemFirst_;   // per entry (+1): members (wheel class only; body-class entries have none)
    std::vector<std::uint32_t> entMembers_;
    std::vector<std::uint8_t> entOverlap_;
    float worldY0_ = 0.0f, worldY1_ = 0.0f;
    struct ColRec { std::uint64_t key; std::uint32_t plate; std::int32_t y0, y1; std::uint8_t box[4]; std::uint32_t tri; };
    void ColumnFrame();
    void ColumnRecords(std::uint32_t tri, std::int32_t ixA, std::int32_t ixB, std::int32_t izA, std::int32_t izB, std::vector<ColRec> *recs) const;
    bool MembersOverlap(std::uint32_t ma, std::uint32_t mb) const;
    struct Emitted {   // EmitSlots' output, any slot count
        std::vector<std::uint32_t> first, unionFirst, entMemFirst, entMembers;
        std::vector<ColumnEntry> entries;
        std::vector<std::int32_t> uni;
        std::vector<std::uint8_t> overlap;
    };
    template <class SlotOf>
    void EmitSlots(const std::vector<ColRec> &recs, SlotOf slotOf, std::size_t nslots, Emitted *out) const;
    void BuildPlates();
    void BuildFootprints();
    void BuildColumns();
};

}  // namespace tw::atlas

#endif
