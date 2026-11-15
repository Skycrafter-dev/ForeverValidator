#include "tw_accel.h"

#include <algorithm>
#include <cmath>

namespace tw {

namespace {

float Area(const Box3 &b) {
    const float dx = b.hi[0] - b.lo[0], dy = b.hi[1] - b.lo[1], dz = b.hi[2] - b.lo[2];
    if (!(dx >= 0.0f)) return 0.0f;
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

struct BinNode {  // binary build node
    Box3 box;
    std::int32_t left = -1, right = -1;  // children, or -1
    std::uint32_t first = 0, count = 0;  // leaf range in the order array
};

struct BinaryBuilder {
    const std::vector<Box3> &boxes;
    std::vector<float> cx, cy, cz;
    std::vector<std::uint32_t> &idx;
    std::vector<BinNode> nodes;

    BinaryBuilder(const std::vector<Box3> &b, std::vector<std::uint32_t> &i) : boxes(b), idx(i) {
        cx.resize(b.size()); cy.resize(b.size()); cz.resize(b.size());
        for (std::size_t k = 0; k < b.size(); ++k) {
            cx[k] = 0.5f * (b[k].lo[0] + b[k].hi[0]);
            cy[k] = 0.5f * (b[k].lo[1] + b[k].hi[1]);
            cz[k] = 0.5f * (b[k].lo[2] + b[k].hi[2]);
        }
    }
    float C(std::uint32_t t, int axis) const { return axis == 0 ? cx[t] : axis == 1 ? cy[t] : cz[t]; }

    std::int32_t Build(std::uint32_t first, std::uint32_t count) {
        BinNode node;
        node.box = EmptyBox();
        Box3 cbox = EmptyBox();
        for (std::uint32_t k = first; k < first + count; ++k) {
            node.box = Union(node.box, boxes[idx[k]]);
            const float c[3] = {cx[idx[k]], cy[idx[k]], cz[idx[k]]};
            for (int a = 0; a < 3; ++a) { cbox.lo[a] = std::min(cbox.lo[a], c[a]); cbox.hi[a] = std::max(cbox.hi[a], c[a]); }
        }
        const std::int32_t self = static_cast<std::int32_t>(nodes.size());
        nodes.push_back(node);
        if (count <= 4u) { nodes[self].first = first; nodes[self].count = count; return self; }
        constexpr int kBins = 16;
        float bestCost = 1e38f; int bestAxis = -1, bestSplit = -1;
        for (int a = 0; a < 3; ++a) {
            const float ext = cbox.hi[a] - cbox.lo[a];
            if (!(ext > 1e-6f)) continue;
            Box3 bb[kBins]; std::uint32_t bc[kBins] = {};
            for (int i = 0; i < kBins; ++i) bb[i] = EmptyBox();
            const float s = kBins / ext;
            for (std::uint32_t k = first; k < first + count; ++k) {
                int bi = static_cast<int>((C(idx[k], a) - cbox.lo[a]) * s);
                bi = std::min(std::max(bi, 0), kBins - 1);
                bb[bi] = Union(bb[bi], boxes[idx[k]]); ++bc[bi];
            }
            Box3 lb[kBins]; std::uint32_t lc[kBins];
            Box3 acc = EmptyBox(); std::uint32_t n = 0;
            for (int i = 0; i < kBins; ++i) { acc = Union(acc, bb[i]); n += bc[i]; lb[i] = acc; lc[i] = n; }
            acc = EmptyBox(); n = 0;
            for (int i = kBins - 1; i > 0; --i) {
                acc = Union(acc, bb[i]); n += bc[i];
                const float cost = Area(lb[i - 1]) * lc[i - 1] + Area(acc) * n;
                if (lc[i - 1] > 0 && n > 0 && cost < bestCost) { bestCost = cost; bestAxis = a; bestSplit = i; }
            }
        }
        std::uint32_t mid;
        if (bestAxis < 0) {
            mid = first + count / 2;
        } else {
            const float ext = cbox.hi[bestAxis] - cbox.lo[bestAxis], s = kBins / ext;
            auto it = std::partition(idx.begin() + first, idx.begin() + first + count, [&](std::uint32_t t) {
                int bi = static_cast<int>((C(t, bestAxis) - cbox.lo[bestAxis]) * s);
                bi = std::min(std::max(bi, 0), kBins - 1);
                return bi < bestSplit;
            });
            mid = static_cast<std::uint32_t>(it - idx.begin());
            if (mid == first || mid == first + count) mid = first + count / 2;
        }
        const std::int32_t l = Build(first, mid - first);
        const std::int32_t r = Build(mid, first + count - mid);
        nodes[self].left = l; nodes[self].right = r;
        return self;
    }
};

}  // namespace

void Bvh4::Build(const World &world) {
    const std::size_t n = world.tris.size();
    std::vector<Box3> boxes(n);
    std::vector<std::uint32_t> idx(n);
    for (std::size_t k = 0; k < n; ++k) {
        for (int a = 0; a < 3; ++a) { boxes[k].lo[a] = world.tris[k].lo[a]; boxes[k].hi[a] = world.tris[k].hi[a]; }
        idx[k] = static_cast<std::uint32_t>(k);
    }
    nodes_.clear(); leaves_.clear();
    if (n == 0) return;
    BinaryBuilder bb(boxes, idx);
    bb.Build(0, static_cast<std::uint32_t>(n));
    order_ = idx;
    boxes_.resize(n);
    for (std::size_t k = 0; k < n; ++k) boxes_[k] = boxes[order_[k]];
    // collapse binary -> 4-wide
    struct Pending { std::int32_t bin; std::int32_t node; };
    std::vector<Pending> work;
    nodes_.push_back(Node{});
    work.push_back({0, 0});
    while (!work.empty()) {
        Pending w = work.back(); work.pop_back();
        std::int32_t kids[4]; int nk = 0;
        const BinNode &root = bb.nodes[w.bin];
        if (root.left < 0) { kids[nk++] = w.bin; }
        else {
            kids[nk++] = root.left; kids[nk++] = root.right;
            while (nk < 4) {  // open the largest internal child
                int best = -1; float ba = -1.0f;
                for (int i = 0; i < nk; ++i) {
                    const BinNode &c = bb.nodes[kids[i]];
                    if (c.left >= 0 && Area(c.box) > ba) { ba = Area(c.box); best = i; }
                }
                if (best < 0) break;
                const BinNode &c = bb.nodes[kids[best]];
                kids[best] = c.left; kids[nk++] = c.right;
            }
        }
        Node node;
        for (int c = 0; c < 4; ++c) {
            for (int a = 0; a < 3; ++a) { node.lo[a][c] = 1e30f; node.hi[a][c] = -1e30f; }
            node.child[c] = -1;  // placeholder; empty slots never overlap
        }
        for (int c = 0; c < nk; ++c) {
            const BinNode &k = bb.nodes[kids[c]];
            for (int a = 0; a < 3; ++a) { node.lo[a][c] = k.box.lo[a]; node.hi[a][c] = k.box.hi[a]; }
            if (k.left < 0) {
                leaves_.push_back({k.first, k.count});
                node.child[c] = -static_cast<std::int32_t>(leaves_.size());
            } else {
                const std::int32_t id = static_cast<std::int32_t>(nodes_.size());
                nodes_.push_back(Node{});
                node.child[c] = id;
                work.push_back({kids[c], id});
            }
        }
        if (nk == 1 && bb.nodes[kids[0]].left < 0) { /* single-leaf root */ }
        nodes_[static_cast<std::size_t>(w.node)] = node;
    }
    // empty slots: point at a zero-count leaf so a stray overlap is harmless
    leaves_.push_back({0u, 0u});
    const std::int32_t emptyLeaf = -static_cast<std::int32_t>(leaves_.size());
    for (Node &nd : nodes_)
        for (int c = 0; c < 4; ++c)
            if (nd.lo[0][c] > nd.hi[0][c]) nd.child[c] = emptyLeaf;
    nodeBox_.assign(nodes_.size(), EmptyBox());
    for (std::size_t i = 0; i < nodes_.size(); ++i)
        for (int c = 0; c < 4; ++c)
            if (nodes_[i].lo[0][c] <= nodes_[i].hi[0][c]) {
                const Box3 cb = {{nodes_[i].lo[0][c], nodes_[i].lo[1][c], nodes_[i].lo[2][c]}, {nodes_[i].hi[0][c], nodes_[i].hi[1][c], nodes_[i].hi[2][c]}};
                nodeBox_[i] = Union(nodeBox_[i], cb);
            }
}

std::size_t Bvh4::Bytes() const {
    return nodes_.size() * sizeof(Node) + nodeBox_.size() * sizeof(Box3) + leaves_.size() * sizeof(Leaf) + order_.size() * 4 + boxes_.size() * sizeof(Box3);
}

void LooseGrid::Build(const World &world, float baseCell) {
    struct Entry { std::uint64_t key; std::uint32_t id; };
    std::vector<Entry> entries;
    entries.reserve(world.tris.size());
    for (int L = 0; L < kLevels; ++L) { cell_[L] = baseCell * static_cast<float>(1 << (2 * L)); levelCount_[L] = 0; }
    hugeBoxes_.clear(); hugeIds_.clear();
    for (std::uint32_t i = 0; i < world.tris.size(); ++i) {
        const Tri &t = world.tris[i];
        const float ext = std::max({t.hi[0] - t.lo[0], t.hi[1] - t.lo[1], t.hi[2] - t.lo[2]});
        int L = 0;
        while (L < kLevels && ext > cell_[L]) ++L;
        if (L == kLevels) {
            Box3 b; for (int a = 0; a < 3; ++a) { b.lo[a] = t.lo[a]; b.hi[a] = t.hi[a]; }
            hugeBoxes_.push_back(b); hugeIds_.push_back(i); continue;
        }
        const float ic = 1.0f / cell_[L];
        const int x = FloorInt(0.5f * (t.lo[0] + t.hi[0]) * ic), y = FloorInt(0.5f * (t.lo[1] + t.hi[1]) * ic),
                  z = FloorInt(0.5f * (t.lo[2] + t.hi[2]) * ic);
        entries.push_back({Key(L, x, y, z), i});
        ++levelCount_[L];
    }
    hugeCount_ = static_cast<std::uint32_t>(hugeIds_.size());
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.key != b.key ? a.key < b.key : a.id < b.id; });
    boxes_.resize(entries.size()); ids_.resize(entries.size());
    std::size_t cells = 0;
    for (std::size_t k = 0; k < entries.size(); ++k) {
        if (k == 0 || entries[k].key != entries[k - 1].key) ++cells;
        const Tri &t = world.tris[entries[k].id];
        for (int a = 0; a < 3; ++a) { boxes_[k].lo[a] = t.lo[a]; boxes_[k].hi[a] = t.hi[a]; }
        ids_[k] = entries[k].id;
    }
    std::size_t cap = 16; while (cap < cells * 2) cap <<= 1;
    slots_.assign(cap, Slot{0, 0, 0});
    mask_ = cap - 1;
    for (std::size_t k = 0; k < entries.size();) {
        std::size_t j = k; while (j < entries.size() && entries[j].key == entries[k].key) ++j;
        std::uint64_t h = Hash(entries[k].key) & mask_;
        while (slots_[h].end != 0u) h = (h + 1) & mask_;
        slots_[h] = {entries[k].key, static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(j)};
        k = j;
    }
}

std::size_t LooseGrid::Bytes() const {
    return slots_.size() * sizeof(Slot) + boxes_.size() * sizeof(Box3) + ids_.size() * 4 + hugeBoxes_.size() * sizeof(Box3) + hugeIds_.size() * 4;
}

}  // namespace tw
