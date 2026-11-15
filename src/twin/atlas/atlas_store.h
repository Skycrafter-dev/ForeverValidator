// Contact atlas: tile store (architect C). Lossless, random-access storage of the atlas's
// foreign-gap tiles and their per-cell member lists, behind the accessors E's fast path calls.
//
// Contract (agreed with E, 2026-11-08):
//   - CellRec (32 B, int16 mm along the plate normal from the plate origin, lo = ceil, hi = floor,
//     clamped to +-32767; classes 0..6, slot 7 = pad lo 32767 / hi -32767) is the canonical value.
//     Every accessor returns exactly the values the tile was built with (bit-identical lookups).
//   - Tiles are keyed (plate, tx, ty), tx = floor(u / 8 m); cell = j * 32 + i (0.25 m cells).
//   - A missing tile returns the always-fail record (every class lo 32767, hi -32767) and no members.
//
// Encoding (per tile, class-major):
//   - uniform tile (all 1024 cells equal): shared record, no per-tile data;
//   - else per class a plane of (lo, hi) pairs: uniform (the pair in the tile header), a palette with
//     4- or 8-bit indices, or raw pairs; classes with equal planes share one;
//   - member lists: u16 per-cell offsets (u32 when a tile holds more than 65535 entries) + u32 tri ids.
// Lookups: uniform tiles and uniform/raw planes are one load after the tile header; palette planes
// add the index load. Each tile is one allocation (header, planes, member lists); (plate, tx, ty)
// resolves through a hash map, once per tick per shape (TileRef), never per lookup.
//
// Residency: a store either holds a whole prebuilt atlas (Add + Seal) or materialises tiles on
// first use through a builder callback, keeping at most `budgetBytes` of encoded tiles (LRU by
// last tick touched). A store is not thread-safe while it materialises; a sealed store is.
#ifndef TW_ATLAS_STORE_H
#define TW_ATLAS_STORE_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "atlas_format.h"   // E's CellRec, kTileCells (the canonical values)

namespace tw::atlas {

constexpr int kCellsPerTile = kTileCells * kTileCells;
constexpr int kClassSlots = 8;                      // classes 0..6, slot 7 pad

// The record of a missing tile (and of slot 7 everywhere): every test fails.
const CellRec &AlwaysFail();

struct MemberSpan {
    const std::uint32_t *data = nullptr;
    std::uint32_t size = 0;
};

// One tile as built (the input of the store): 1024 records and the per-cell member lists.
struct RawTile {
    CellRec cell[kCellsPerTile];
    std::vector<std::uint32_t> first;     // kCellsPerTile + 1 offsets into members (or empty: no members)
    std::vector<std::uint32_t> members;   // world triangle ids
};

struct EncodedTile;   // atlas_store.cpp

// Resolved per tick per shape; cheap to copy.
struct TileRef {
    const EncodedTile *t = nullptr;   // null: missing tile
};

struct StoreStats {
    std::size_t tiles = 0, uniformTiles = 0, rawBytes = 0, encodedBytes = 0, memberRawBytes = 0, memberBytes = 0, indexBytes = 0;
    std::size_t planes[4] = {};       // per class plane: uniform, 4-bit, 8-bit, raw
    std::size_t sharedPlanes = 0;     // planes equal to an earlier class's plane in the same tile
    std::size_t materialised = 0, evicted = 0, residentBytes = 0, peakResidentBytes = 0;
};

class Store {
public:
    Store();
    ~Store();
    Store(const Store &) = delete;
    Store &operator=(const Store &) = delete;

    // ---- building ----
    // Prebuilt atlas: add every tile, then Seal(). plateCount bounds the plate ids.
    void Reset(std::uint32_t plateCount);
    void Add(std::uint32_t plate, std::int32_t tx, std::int32_t ty, const RawTile &tile);
    void Seal();
    // Lazy atlas: tiles are built on first use by `build` (true: tile exists), keeping at most
    // budgetBytes of encoded tiles resident (0: no limit). Tick() advances the LRU clock.
    using Builder = std::function<bool(std::uint32_t plate, std::int32_t tx, std::int32_t ty, RawTile *out)>;
    void SetBuilder(Builder build, std::size_t budgetBytes);
    void Tick() { ++clock_; }

    // ---- lookups (E's fast path) ----
    TileRef Tile(std::uint32_t plate, std::int32_t tx, std::int32_t ty);
    // The whole record (diagnostics): a pointer into the tile for uniform/raw tiles, else to a
    // per-thread decode buffer valid until the next call.
    const CellRec *Cell(TileRef t, int cell) const;
    // lo/hi of class cls[i] at cell[i] of tile t[i], i < n.
    void GatherClass(const TileRef *t, const int *cell, const int *cls, int n, std::int16_t *lo, std::int16_t *hi) const;
    MemberSpan Members(TileRef t, int cell) const;

    const StoreStats &Stats() const { return stats_; }
    // Bit-identity check of one tile against its raw form (tests).
    bool Equal(TileRef t, const RawTile &raw) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    StoreStats stats_;
    std::uint64_t clock_ = 0;
};

}  // namespace tw::atlas

#endif
