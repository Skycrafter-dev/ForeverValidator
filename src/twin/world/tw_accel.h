// Static acceleration structures over the triangle soup: BVH4 and a loose
// multi-level hashed grid. Both answer "triangles whose AABB overlaps a box";
// the result is exact (no false negatives by construction: closed-box tests).
#ifndef TW_ACCEL_H
#define TW_ACCEL_H

#include <cstdint>
#include <vector>

#include "tw_world.h"

namespace tw {

class Bvh4 {
public:
    void Build(const World &world);
    std::size_t Bytes() const;
    std::size_t NodeCount() const { return nodes_.size(); }

    // Deepest node whose box contains b (start for coherent queries); 0 = root.
    std::int32_t DeepestContaining(const Box3 &b) const {
        std::int32_t n = 0;
        for (;;) {
            const Node &nd = nodes_[static_cast<std::uint32_t>(n)];
            std::int32_t next = -1;
            for (int c = 0; c < 4; ++c) {
                if (nd.child[c] < 0) continue;
                const bool in = nd.lo[0][c] <= b.lo[0] && nd.lo[1][c] <= b.lo[1] && nd.lo[2][c] <= b.lo[2] && nd.hi[0][c] >= b.hi[0] &&
                                nd.hi[1][c] >= b.hi[1] && nd.hi[2][c] >= b.hi[2];
                if (in) { next = nd.child[c]; break; }
            }
            if (next < 0) return n;
            n = next;
        }
    }
    const Box3 &NodeBox(std::int32_t n) const { return nodeBox_[static_cast<std::uint32_t>(n)]; }
    // Calls f(triId) for every triangle whose AABB overlaps q. A non-root start is NOT exact for a
    // BVH (sibling boxes overlap); it exists only for experiments.
    template <class F>
    void Query(const Box3 &q, F &&f, std::int32_t start = 0) const {
        if (nodes_.empty()) return;
        std::int32_t stack[96];
        int sp = 0;
        stack[sp++] = start;
        while (sp > 0) {
            const Node &n = nodes_[static_cast<std::uint32_t>(stack[--sp])];
            for (int c = 0; c < 4; ++c) {
                const bool ov = n.lo[0][c] <= q.hi[0] && q.lo[0] <= n.hi[0][c] && n.lo[1][c] <= q.hi[1] &&
                                q.lo[1] <= n.hi[1][c] && n.lo[2][c] <= q.hi[2] && q.lo[2] <= n.hi[2][c];
                if (!ov) continue;
                const std::int32_t ch = n.child[c];
                if (ch >= 0) {
                    if (sp < 96) stack[sp++] = ch;
                } else {
                    const Leaf &lf = leaves_[static_cast<std::uint32_t>(-ch - 1)];
                    for (std::uint32_t k = lf.first; k < lf.first + lf.count; ++k)
                        if (Overlap(boxes_[k], q)) f(order_[k]);
                }
            }
        }
    }

private:
    struct Node {
        float lo[3][4], hi[3][4];
        std::int32_t child[4];  // >= 0 node, < 0 leaf -(index + 1)
    };
    struct Leaf { std::uint32_t first, count; };
    std::vector<Node> nodes_;
    std::vector<Box3> nodeBox_;
    std::vector<Leaf> leaves_;
    std::vector<std::uint32_t> order_;  // triangle ids in BVH leaf order
    std::vector<Box3> boxes_;           // their AABBs in the same order
};

// Loose grid: each triangle is stored once, in the cell of its AABB centre,
// at the smallest level whose cell is at least its largest extent. A query
// visits, per level, the cells overlapping the box inflated by half a cell.
class LooseGrid {
public:
    static constexpr int kLevels = 6;
    void Build(const World &world, float baseCell = 4.0f);
    std::size_t Bytes() const;

    template <class F>
    void Query(const Box3 &q, F &&f) const {
        for (int L = 0; L < kLevels; ++L) {
            if (levelCount_[L] == 0u) continue;
            const float c = cell_[L], ic = 1.0f / c, hc = 0.5f * c;
            int lo[3], hi[3];
            for (int k = 0; k < 3; ++k) {
                lo[k] = FloorInt((q.lo[k] - hc) * ic);
                hi[k] = FloorInt((q.hi[k] + hc) * ic);
            }
            for (int x = lo[0]; x <= hi[0]; ++x)
                for (int y = lo[1]; y <= hi[1]; ++y)
                    for (int z = lo[2]; z <= hi[2]; ++z) {
                        std::uint32_t b, e;
                        if (!Find(L, x, y, z, &b, &e)) continue;
                        for (std::uint32_t k = b; k < e; ++k)
                            if (Overlap(boxes_[k], q)) f(ids_[k]);
                    }
        }
        for (std::uint32_t k = 0; k < hugeCount_; ++k)
            if (Overlap(hugeBoxes_[k], q)) f(hugeIds_[k]);
    }

private:
    static int FloorInt(float v) { const int i = static_cast<int>(v); return (static_cast<float>(i) > v) ? i - 1 : i; }
    static std::uint64_t Key(int L, int x, int y, int z) {
        return (static_cast<std::uint64_t>(L) << 60) ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x) & 0xfffffu) << 40) ^
               (static_cast<std::uint64_t>(static_cast<std::uint32_t>(y) & 0xfffffu) << 20) ^
               static_cast<std::uint64_t>(static_cast<std::uint32_t>(z) & 0xfffffu);
    }
    static std::uint64_t Hash(std::uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return k; }
    bool Find(int L, int x, int y, int z, std::uint32_t *b, std::uint32_t *e) const {
        const std::uint64_t key = Key(L, x, y, z);
        std::uint64_t h = Hash(key) & mask_;
        for (;;) {
            const Slot &s = slots_[h];
            if (s.end == 0u) return false;
            if (s.key == key) { *b = s.begin; *e = s.end; return true; }
            h = (h + 1) & mask_;
        }
    }
    struct Slot { std::uint64_t key; std::uint32_t begin, end; };
    float cell_[kLevels] = {};
    std::uint32_t levelCount_[kLevels] = {};
    std::vector<Slot> slots_;
    std::uint64_t mask_ = 0;
    std::vector<Box3> boxes_;
    std::vector<std::uint32_t> ids_;
    std::vector<Box3> hugeBoxes_;
    std::vector<std::uint32_t> hugeIds_;
    std::uint32_t hugeCount_ = 0u;
};

}  // namespace tw

#endif
