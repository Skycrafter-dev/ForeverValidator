// Contact atlas: packed column grid (architect C). The 2 m x-z column grid of atlas_build.h (E) in a
// compact, read-only layout that returns E's values bit for bit: per (column, class) slot the entries
// (plate, y0, y1 in integer mm, the prefix max y1max, the u8 x-z box), their union (the free test)
// and, for wheel-class entries, the member lists and overlap flags.
//
// Layout: 16 x 16-column superblocks. Each holds a table of the plates its slots name (entries carry a
// u8 or u16 local index), u16 slot offsets into its payload (u32 when the payload exceeds 64 KB), and
// the payloads; identical payloads within a superblock are stored once. A payload (fixed width, SoA):
//   n (u8, 255 -> u32), flags (bit 0: 32-bit y fields, bit 1: some boxes are not the whole column),
//   base = the first entry's y0 (int32 mm), then y0 - base, y1max - base and y1 - y0 (u16 each, or u32),
//   the local plate indices, a box mask and the non-full boxes (4 B), then per entry the varint member
//   count, and if non-zero the overlap byte and the members as plate-local u16 (u32 for loose entries and
//   plates of more than 65536 members).
// The union is derived from the decoded entries with E's own merge (atlas_build.cpp), so it is
// bit-identical too; the free test (AnyOverlap) uses the stored prefix max instead of the union: the first
// entry whose y1max reaches lo decides it.
#ifndef TW_ATLAS_COLPACK_H
#define TW_ATLAS_COLPACK_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "atlas_build.h"

namespace tw::atlas {

// E's raw grid, read-only (atlas_build.h Atlas::BuildColumns).
struct RawColumns {
    float x0 = 0, z0 = 0;
    std::uint32_t nx = 0, nz = 0;
    const std::uint32_t *first = nullptr;        // 2 * nx * nz + 1
    const ColumnEntry *entries = nullptr;
    const std::uint32_t *entMemFirst = nullptr;  // per entry + 1
    const std::uint32_t *entMembers = nullptr;   // Atlas::Members() indices
    const std::uint8_t *entOverlap = nullptr;
    const PlateRec *plates = nullptr;
    std::size_t plateCount = 0;
};

class ColumnPack {
public:
    static constexpr int kSuper = 16;            // columns per superblock edge

    void Build(const RawColumns &raw);
    std::size_t Bytes() const;

    // E's ColumnSlot formula (E's slot numbering: (iz * nx + ix) * 2 + cls).
    std::uint32_t ColumnSlot(const float x[3], int cls) const {
        const float fx = (x[0] - x0_) * (1.0f / kColumnM), fz = (x[2] - z0_) * (1.0f / kColumnM);
        if (!(fx >= 0.0f && fz >= 0.0f)) return 0xFFFFFFFFu;
        const std::uint32_t ix = static_cast<std::uint32_t>(fx), iz = static_cast<std::uint32_t>(fz);
        if (ix >= nx_ || iz >= nz_) return 0xFFFFFFFFu;
        return (iz * nx_ + ix) * 2u + static_cast<std::uint32_t>(cls);
    }
    // The same slot as a handle for the lookups below without the division by nx: superblock << 9 | local slot.
    std::uint32_t SlotHandle(const float x[3], int cls) const {
        const float fx = (x[0] - x0_) * (1.0f / kColumnM), fz = (x[2] - z0_) * (1.0f / kColumnM);
        if (!(fx >= 0.0f && fz >= 0.0f)) return 0xFFFFFFFFu;
        const std::uint32_t ix = static_cast<std::uint32_t>(fx), iz = static_cast<std::uint32_t>(fz);
        if (ix >= nx_ || iz >= nz_) return 0xFFFFFFFFu;
        return ((iz / kSuper) * sx_ + ix / kSuper) << 9 | (((iz % kSuper) * kSuper + (ix % kSuper)) * 2u + static_cast<std::uint32_t>(cls));
    }
    std::uint32_t HandleOf(std::uint32_t slot) const {   // E's slot number -> handle
        const std::uint32_t col = slot >> 1, ix = col % nx_, iz = col / nx_;
        return ((iz / kSuper) * sx_ + ix / kSuper) << 9 | (((iz % kSuper) * kSuper + (ix % kSuper)) * 2u + (slot & 1u));
    }

    // A decoded slot (valid until the next Decode into the same view).
    struct SlotView {
        std::uint32_t n = 0, unionCount = 0;
        std::vector<ColumnEntry> entries;           // E's records, y1max included
        std::vector<std::int32_t> unionYs;          // y0, y1 pairs (mm)
        std::vector<std::uint32_t> memFirst;        // n + 1
        std::vector<std::uint32_t> members;         // Atlas::Members() indices
        std::vector<std::uint8_t> overlap;          // per entry
    };
    // handle: SlotHandle / HandleOf. what: 2 union, 4 members (entries are always decoded)
    void Decode(std::uint32_t handle, SlotView *out, int what = 6) const;
    // The free test alone: does any entry's y-range meet [lo, hi] (mm)? (E's union test, without a decode)
    bool AnyOverlap(std::uint32_t handle, std::int32_t lo, std::int32_t hi) const;
    const void *PayloadPtr(std::uint32_t handle) const { const Super *sb; return Payload(handle, &sb); }   // measurement

private:
    struct Super {
        std::uint32_t data = 0;        // payload base in bytes_ (slot offsets first)
        std::uint32_t plates = 0;      // first local plate in plateIds_
        std::int32_t yBase = 0;        // y0 of the superblock's first entry (mm)
        std::uint16_t plateCount = 0;
        std::uint8_t wideIdx = 0;      // local plate indices are u16
        std::uint8_t wideOff = 0;      // slot offsets are u32
    };
    float x0_ = 0, z0_ = 0;
    std::uint32_t nx_ = 0, nz_ = 0, sx_ = 0, sz_ = 0;
    std::vector<Super> supers_;
    std::vector<std::uint32_t> plateIds_;
    std::vector<std::uint8_t> bytes_;
    std::vector<std::uint32_t> plateFirstMember_;
    std::vector<std::uint8_t> plateWideMembers_;
    const std::uint8_t *Payload(std::uint32_t handle, const Super **sb) const;
};

}  // namespace tw::atlas

#endif
