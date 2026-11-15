#include "atlas_build.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>

#include "tw_exact.h"

namespace tw::atlas {

namespace {

struct D3 { double x, y, z; };
inline D3 Sub(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline D3 Add(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline D3 Mul(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double Dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline D3 Cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double Len(D3 a) { return std::sqrt(Dot(a, a)); }
inline D3 V(const float *p) { return {p[0], p[1], p[2]}; }
inline D3 Vd(const double *p) { return {p[0], p[1], p[2]}; }

double MsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// clip the triangle (u, v, h) to [u0, u1] x [v0, v1]; min / max h of the clipped polygon
bool ClipHeights(const D3 *tri, double u0, double u1, double v0, double v1, double *a, double *b) {
    D3 buf0[16], buf1[16];
    int n = 3;
    for (int k = 0; k < 3; ++k) buf0[k] = tri[k];
    D3 *in = buf0, *out = buf1;
    for (int pl = 0; pl < 4; ++pl) {
        int m = 0;
        for (int k = 0; k < n; ++k) {
            const D3 &p = in[k], &q = in[(k + 1) % n];
            auto f = [&](const D3 &x) {
                switch (pl) { case 0: return x.x - u0; case 1: return u1 - x.x; case 2: return x.y - v0; default: return v1 - x.y; }
            };
            const double fp = f(p), fq = f(q);
            if (fp >= 0) out[m++] = p;
            if ((fp >= 0) != (fq >= 0)) {
                const double t = fp / (fp - fq);
                out[m++] = {p.x + t * (q.x - p.x), p.y + t * (q.y - p.y), p.z + t * (q.z - p.z)};
            }
        }
        n = m;
        if (n == 0) return false;
        std::swap(in, out);
    }
    double lo = 1e30, hi = -1e30;
    for (int k = 0; k < n; ++k) { lo = std::min(lo, in[k].z); hi = std::max(hi, in[k].z); }
    *a = lo; *b = hi;
    return true;
}

// clip the triangle (u, v, h) to the rectangle; the clipped polygon's bounds in u, v and h
bool ClipBounds(const D3 *tri, double u0, double u1, double v0, double v1, double b[6]) {
    D3 buf0[16], buf1[16];
    int n = 3;
    for (int k = 0; k < 3; ++k) buf0[k] = tri[k];
    D3 *in = buf0, *out = buf1;
    for (int pl = 0; pl < 4; ++pl) {
        int m = 0;
        for (int k = 0; k < n; ++k) {
            const D3 &p = in[k], &q = in[(k + 1) % n];
            auto f = [&](const D3 &x) {
                switch (pl) { case 0: return x.x - u0; case 1: return u1 - x.x; case 2: return x.y - v0; default: return v1 - x.y; }
            };
            const double fp = f(p), fq = f(q);
            if (fp >= 0) out[m++] = p;
            if ((fp >= 0) != (fq >= 0)) {
                const double t = fp / (fp - fq);
                out[m++] = {p.x + t * (q.x - p.x), p.y + t * (q.y - p.y), p.z + t * (q.z - p.z)};
            }
        }
        n = m;
        if (n == 0) return false;
        std::swap(in, out);
    }
    b[0] = b[2] = b[4] = 1e30; b[1] = b[3] = b[5] = -1e30;
    for (int k = 0; k < n; ++k) {
        b[0] = std::min(b[0], in[k].x); b[1] = std::max(b[1], in[k].x);
        b[2] = std::min(b[2], in[k].y); b[3] = std::max(b[3], in[k].y);
        b[4] = std::min(b[4], in[k].z); b[5] = std::max(b[5], in[k].z);
    }
    return true;
}

inline std::int32_t FloorDiv(double x, double s) { return static_cast<std::int32_t>(std::floor(x / s)); }

std::uint64_t TileKey64(std::int32_t tx, std::int32_t ty) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(tx)) << 32) | static_cast<std::uint32_t>(ty);
}


}  // namespace

bool Atlas::Build(const world::WorldPackView &view, const Bvh4 &bvh, const BuildOptions &opt, std::string *error) {
    const auto t0 = std::chrono::steady_clock::now();
    view_ = &view;
    bvh_ = &bvh;
    if (view.tris.size() >= 0xFFFFFFF0u) { if (error) *error = "atlas: too many triangles"; return false; }
    BuildPlates();
    BuildFootprints();
    stats.buildMs = MsSince(t0);
    const auto t1 = std::chrono::steady_clock::now();
    if (opt.lazyColumns) ColumnFrame(); else BuildColumns();
    stats.gridMs = MsSince(t1);
    if (opt.allTiles) {
        for (std::uint32_t p = 0; p < plates_.size(); ++p)
            for (std::uint64_t k : footprint_[p]) (void)TileAt(p, static_cast<std::int32_t>(k >> 32), static_cast<std::int32_t>(k & 0xFFFFFFFFu));
    }
    return true;
}

void Atlas::BuildPlates() {
    const world::WorldPackView &v = *view_;
    const std::size_t nt = v.tris.size();
    plateOf_.assign(nt, kNoPlate);
    memberOf_.assign(nt, kNoPlate);
    triN_.assign(nt * 4, 0.0);
    loose_.assign(nt, 0);
    stats.tris = nt;
    // exact duplicates (the same vertex set): FV emits one FACE per copy, so they are loose
    {
        struct VKey { float v[9]; std::uint32_t tri; };
        std::vector<VKey> keys(nt);
        for (std::size_t i = 0; i < nt; ++i) {
            const Tri &t = v.tris[i];
            int idx[3] = {0, 1, 2};
            std::sort(idx, idx + 3, [&](int p, int q) { return std::memcmp(t.v[p], t.v[q], 12) < 0; });
            for (int j = 0; j < 3; ++j) std::memcpy(keys[i].v + 3 * j, t.v[idx[j]], 12);
            keys[i].tri = static_cast<std::uint32_t>(i);
        }
        std::sort(keys.begin(), keys.end(), [](const VKey &a, const VKey &b) {
            const int c = std::memcmp(a.v, b.v, sizeof a.v);
            return c != 0 ? c < 0 : a.tri < b.tri;
        });
        for (std::size_t i = 1; i < nt; ++i)
            if (std::memcmp(keys[i].v, keys[i - 1].v, sizeof keys[i].v) == 0) {
                if (!loose_[keys[i].tri]) ++stats.dups;
                loose_[keys[i].tri] = 1; loose_[keys[i - 1].tri] = 1;
            }
    }
    struct Key { std::int64_t q[4]; std::uint32_t tri; };
    std::vector<Key> keys;
    keys.reserve(nt);
    for (std::size_t i = 0; i < nt; ++i) {
        const Tri &t = v.tris[i];
        const D3 v0 = V(t.v[0]), v1 = V(t.v[1]), v2 = V(t.v[2]);
        D3 n = Cross(Sub(v1, v0), Sub(v2, v0));
        const double l = Len(n);
        const bool trig = (t.meta & kMetaTrigger) != 0 || v.triInfo[i].trigger != world::kNoTrigger;
        if (!(l > 1e-9) || trig) { loose_[i] = 1; continue; }
        n = Mul(n, 1.0 / l);
        double *tn = &triN_[4 * i];
        tn[0] = n.x; tn[1] = n.y; tn[2] = n.z; tn[3] = Dot(n, v0);
        if (loose_[i]) continue;
        Key k;
        k.q[0] = std::llround(n.x / kPlateAngle); k.q[1] = std::llround(n.y / kPlateAngle); k.q[2] = std::llround(n.z / kPlateAngle);
        k.q[3] = std::llround(tn[3] / kPlateOffset);
        k.tri = static_cast<std::uint32_t>(i);
        keys.push_back(k);
    }
    std::sort(keys.begin(), keys.end(), [](const Key &a, const Key &b) {
        for (int j = 0; j < 4; ++j) if (a.q[j] != b.q[j]) return a.q[j] < b.q[j];
        return a.tri < b.tri;
    });
    // greedy clustering of consecutive keys within tolerance of the cluster's first member
    std::vector<std::uint32_t> mem;
    std::size_t i = 0;
    while (i < keys.size()) {
        const std::uint32_t t0 = keys[i].tri;
        const double *n0 = &triN_[4 * t0];
        mem.clear();
        std::size_t j = i;
        while (j < keys.size()) {
            const std::uint32_t tj = keys[j].tri;
            const double *nj = &triN_[4 * tj];
            const double dn = std::sqrt((nj[0] - n0[0]) * (nj[0] - n0[0]) + (nj[1] - n0[1]) * (nj[1] - n0[1]) + (nj[2] - n0[2]) * (nj[2] - n0[2]));
            if (dn > 2 * kPlateAngle || std::fabs(nj[3] - n0[3]) > 2 * kPlateOffset) break;
            mem.push_back(tj);
            ++j;
        }
        if (j == i) { mem.push_back(t0); j = i + 1; }
        const std::uint32_t id = static_cast<std::uint32_t>(plates_.size());
        PlateRec p{};
        const Tri &tt = v.tris[t0];
        const D3 o = V(tt.v[0]);
        const D3 n = {n0[0], n0[1], n0[2]};
        const D3 ref = std::fabs(n.x) < 0.6 ? D3{1, 0, 0} : D3{0, 1, 0};
        D3 e1 = Cross(ref, n); e1 = Mul(e1, 1.0 / Len(e1));
        const D3 e2 = Cross(n, e1);
        p.o[0] = o.x; p.o[1] = o.y; p.o[2] = o.z;
        p.n[0] = float(n.x); p.n[1] = float(n.y); p.n[2] = float(n.z);
        p.e1[0] = float(e1.x); p.e1[1] = float(e1.y); p.e1[2] = float(e1.z);
        p.e2[0] = float(e2.x); p.e2[1] = float(e2.y); p.e2[2] = float(e2.z);
        double tau = 0, alpha = 0;
        p.firstMember = static_cast<std::uint32_t>(members_.size());
        for (std::uint32_t m : mem) {
            const Tri &tm = v.tris[m];
            const double *nm = &triN_[4 * m];
            D3 L[3];
            for (int k = 0; k < 3; ++k) {
                const D3 d = Sub(V(tm.v[k]), o);
                L[k] = {Dot(d, e1), Dot(d, e2), Dot(d, n)};
                tau = std::max(tau, std::fabs(L[k].z));
            }
            alpha = std::max(alpha, std::sqrt((nm[0] - n.x) * (nm[0] - n.x) + (nm[1] - n.y) * (nm[1] - n.y) + (nm[2] - n.z) * (nm[2] - n.z)));
            MemberRec r{};
            const double area = (L[1].x - L[0].x) * (L[2].y - L[0].y) - (L[1].y - L[0].y) * (L[2].x - L[0].x);
            const double sg = area >= 0 ? 1.0 : -1.0;
            for (int e = 0; e < 3; ++e) {
                const D3 &a = L[e], &b = L[(e + 1) % 3];
                const double ex = b.x - a.x, ey = b.y - a.y, el = std::sqrt(ex * ex + ey * ey);
                if (!(el > 1e-9)) { r.ln[e][0] = 0; r.ln[e][1] = 0; r.ln[e][2] = -1; continue; }   // degenerate edge: never inside
                r.ln[e][0] = float(-sg * ey / el);
                r.ln[e][1] = float(sg * ex / el);
                r.ln[e][2] = float(sg * (ey * a.x - ex * a.y) / el);
            }
            r.n[0] = float(nm[0]); r.n[1] = float(nm[1]); r.n[2] = float(nm[2]);
            r.c = float(nm[3] - (nm[0] * o.x + nm[1] * o.y + nm[2] * o.z));
            r.tri = m;
            r.plate = id;
            plateOf_[m] = id;
            memberOf_[m] = static_cast<std::uint32_t>(members_.size());
            members_.push_back(r);
        }
        p.memberCount = static_cast<std::uint32_t>(members_.size()) - p.firstMember;
        p.tau = float(tau);
        p.alpha = float(alpha);
        p.slack = float(tau + 2.5 * alpha + 2e-4);
        // tan(30 deg - delta), delta = 2 alpha + 2e-3, without libm (the object audit allows no inexact libm call):
        // (1/sqrt3 - tan delta) / (1 + tan delta / sqrt3), tan delta by its series (delta <= ~0.01: error < 1e-15), rounded down
        const double delta = 2.0 * alpha + 2e-3;
        const double td = delta + delta * delta * delta / 3.0 + 2.0 * delta * delta * delta * delta * delta / 15.0;
        const double t30 = 1.0 / std::sqrt(3.0);
        const double tl = delta < 0.5 ? (t30 - td) / (1.0 + td * t30) : 0.0;
        p.tanProof = tl > 0.0 ? float(tl * (1.0 - 1e-6)) : 0.0f;
        plates_.push_back(p);
        i = j;
    }
    for (auto l : loose_) stats.loose += l;
    stats.plates = plates_.size();
    stats.members = members_.size();
}

void Atlas::BuildFootprints() {
    footprint_.assign(plates_.size(), {});
    const world::WorldPackView &v = *view_;
    const double border = kTileBorderCells * double(kCellM);
    for (std::uint32_t p = 0; p < plates_.size(); ++p) {
        const PlateRec &P = plates_[p];
        const D3 o = Vd(P.o), e1 = V(P.e1), e2 = V(P.e2);
        std::vector<std::uint64_t> &fp = footprint_[p];
        for (std::uint32_t m = P.firstMember; m < P.firstMember + P.memberCount; ++m) {
            const Tri &t = v.tris[members_[m].tri];
            double u0 = 1e30, u1 = -1e30, v0 = 1e30, v1 = -1e30;
            for (int k = 0; k < 3; ++k) {
                const D3 d = Sub(V(t.v[k]), o);
                const double u = Dot(d, e1), w = Dot(d, e2);
                u0 = std::min(u0, u); u1 = std::max(u1, u); v0 = std::min(v0, w); v1 = std::max(v1, w);
            }
            const std::int32_t tx0 = FloorDiv(u0 - border, kTileM), tx1 = FloorDiv(u1 + border, kTileM);
            const std::int32_t ty0 = FloorDiv(v0 - border, kTileM), ty1 = FloorDiv(v1 + border, kTileM);
            for (std::int32_t ty = ty0; ty <= ty1; ++ty)
                for (std::int32_t tx = tx0; tx <= tx1; ++tx) fp.push_back(TileKey64(tx, ty));
        }
        std::sort(fp.begin(), fp.end());
        fp.erase(std::unique(fp.begin(), fp.end()), fp.end());
        fp.shrink_to_fit();
    }
}

bool Atlas::TileHasFootprint(std::uint32_t plate, std::int32_t tx, std::int32_t ty) const {
    const std::vector<std::uint64_t> &fp = footprint_[plate];
    return std::binary_search(fp.begin(), fp.end(), TileKey64(tx, ty));
}

const Tile *Atlas::TileAt(std::uint32_t plate, std::int32_t tx, std::int32_t ty) const {
    if (plate >= plates_.size()) return nullptr;
    const TileKey key{plate, tx, ty};
    {
        std::lock_guard<std::mutex> lock(tileMutex_);
        auto it = tiles_.find(key);
        if (it != tiles_.end()) return it->second.get();
    }
    if (!TileHasFootprint(plate, tx, ty)) return nullptr;
    std::unique_ptr<Tile> t = BuildTile(plate, tx, ty);   // outside the lock; deterministic
    std::lock_guard<std::mutex> lock(tileMutex_);
    auto ins = tiles_.emplace(key, std::move(t));
    return ins.first->second.get();
}

std::size_t Atlas::TilesBuilt() const {
    std::lock_guard<std::mutex> lock(tileMutex_);
    return tiles_.size();
}

std::size_t Atlas::Bytes() const {
    std::size_t b = plates_.size() * sizeof(PlateRec) + members_.size() * sizeof(MemberRec) + colFirst_.size() * 4 +
                    colEntries_.size() * sizeof(ColumnEntry) + colUnionFirst_.size() * 4 + colUnion_.size() * 4 + entMemFirst_.size() * 4 +
                    entMembers_.size() * 4 + entOverlap_.size();
    for (const auto &f : footprint_) b += f.size() * 8;
    std::lock_guard<std::mutex> lock(tileMutex_);
    for (const auto &kv : tiles_) b += sizeof(Tile) + kv.second->memFirst.size() * 4 + kv.second->memList.size() * 4;
    return b;
}

std::unique_ptr<Tile> Atlas::BuildTile(std::uint32_t pid, std::int32_t tx, std::int32_t ty) const {
    const world::WorldPackView &v = *view_;
    const PlateRec &P = plates_[pid];
    const D3 o = Vd(P.o), n = V(P.n), e1 = V(P.e1), e2 = V(P.e2);
    std::unique_ptr<Tile> T(new Tile);
    constexpr int RN = kTileCells + 2 * kTileBorderCells;
    std::vector<float> rlo(RN * RN, -100.0f), rhi(RN * RN, 100.0f);
    const double cell = kCellM, tile = kTileM;
    const double U0 = tx * tile - kTileBorderCells * cell, V0 = ty * tile - kTileBorderCells * cell;
    const double U1 = U0 + RN * cell, V1 = V0 + RN * cell;
    const double cu0 = tx * tile, cv0 = ty * tile;
    // back-face exclusion column: the tile dilated by kBackDilation, centre heights [kCentreLo, kCentreHi]
    D3 corners[8];
    int ci = 0;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c)
                corners[ci++] = Add(Add(o, Add(Mul(e1, cu0 - kBackDilation + a * (tile + 2 * kBackDilation)),
                                               Mul(e2, cv0 - kBackDilation + b * (tile + 2 * kBackDilation)))),
                                    Mul(n, c ? kCentreHi : kCentreLo));
    Box3 q = EmptyBox();
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c) {
                const D3 x = Add(Add(o, Add(Mul(e1, a ? U1 : U0), Mul(e2, b ? V1 : V0))), Mul(n, c ? kFieldHi : kFieldLo));
                const float f[3] = {static_cast<float>(x.x), static_cast<float>(x.y), static_cast<float>(x.z)};
                for (int k = 0; k < 3; ++k) { q.lo[k] = std::min(q.lo[k], f[k] - 0.01f); q.hi[k] = std::max(q.hi[k], f[k] + 0.01f); }
            }
    std::vector<std::vector<std::uint32_t>> cellm(kTileCells * kTileCells);
    bvh_->Query(q, [&](std::uint32_t ti) {
        const Tri &t = v.tris[ti];
        D3 L[3];
        for (int k = 0; k < 3; ++k) {
            const D3 d = Sub(V(t.v[k]), o);
            L[k] = {Dot(d, e1), Dot(d, e2), Dot(d, n)};
        }
        const double bu0 = std::min({L[0].x, L[1].x, L[2].x}), bu1 = std::max({L[0].x, L[1].x, L[2].x});
        const double bv0 = std::min({L[0].y, L[1].y, L[2].y}), bv1 = std::max({L[0].y, L[1].y, L[2].y});
        if (plateOf_[ti] == pid) {   // member: per-cell lists over the tile
            const int i0 = std::max(0, FloorDiv(bu0 - 1e-3 - cu0, cell)), i1 = std::min(kTileCells - 1, FloorDiv(bu1 + 1e-3 - cu0, cell));
            const int j0 = std::max(0, FloorDiv(bv0 - 1e-3 - cv0, cell)), j1 = std::min(kTileCells - 1, FloorDiv(bv1 + 1e-3 - cv0, cell));
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) cellm[j * kTileCells + i].push_back(memberOf_[ti]);
            return;
        }
        const double *tn = &triN_[4 * ti];
        if (tn[0] != 0 || tn[1] != 0 || tn[2] != 0) {
            bool allBehind = true;
            for (int k = 0; k < 8 && allBehind; ++k) allBehind = tn[0] * corners[k].x + tn[1] * corners[k].y + tn[2] * corners[k].z - tn[3] < -1e-3;
            if (allBehind) return;
        }
        const int i0 = std::max(0, FloorDiv(bu0 - U0, cell)), i1 = std::min(RN - 1, FloorDiv(bu1 - U0, cell));
        const int j0 = std::max(0, FloorDiv(bv0 - V0, cell)), j1 = std::min(RN - 1, FloorDiv(bv1 - V0, cell));
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i) {
                double ha, hb;
                if (!ClipHeights(L, U0 + i * cell, U0 + (i + 1) * cell, V0 + j * cell, V0 + (j + 1) * cell, &ha, &hb)) continue;
                float &lo = rlo[j * RN + i], &hi = rhi[j * RN + i];
                if (ha < kHref) lo = std::max(lo, static_cast<float>(hb + 1e-4));
                if (hb > kHref) hi = std::min(hi, static_cast<float>(ha - 1e-4));
            }
    });
    // dilation per class: separable max (lo) / min (hi) filters of radius ceil(R / cell) cells
    std::vector<float> tlo(RN * RN), thi(RN * RN);
    for (int k = 0; k < 8; ++k) {
        if (k >= kClasses) {
            for (int c = 0; c < kTileCells * kTileCells; ++c) { T->cell[c].lo[k] = kGapNone; T->cell[c].hi[k] = -kGapNone; }
            continue;
        }
        const int r = static_cast<int>(std::ceil(double(kClassR[k]) / cell - 1e-9));
        for (int j = 0; j < RN; ++j)
            for (int i = 0; i < RN; ++i) {
                float lo = -100, hi = 100;
                for (int d = -r; d <= r; ++d) {
                    const int ii = i + d;
                    if (ii < 0 || ii >= RN) continue;
                    lo = std::max(lo, rlo[j * RN + ii]); hi = std::min(hi, rhi[j * RN + ii]);
                }
                tlo[j * RN + i] = lo; thi[j * RN + i] = hi;
            }
        for (int j = 0; j < kTileCells; ++j)
            for (int i = 0; i < kTileCells; ++i) {
                float lo = -100, hi = 100;
                for (int d = -r; d <= r; ++d) {
                    const int jj = j + kTileBorderCells + d;
                    lo = std::max(lo, tlo[jj * RN + i + kTileBorderCells]); hi = std::min(hi, thi[jj * RN + i + kTileBorderCells]);
                }
                const double l = std::ceil(double(lo) * 1000.0), h = std::floor(double(hi) * 1000.0);
                T->cell[j * kTileCells + i].lo[k] = static_cast<std::int16_t>(std::max(-32767.0, std::min(32767.0, l)));
                T->cell[j * kTileCells + i].hi[k] = static_cast<std::int16_t>(std::max(-32767.0, std::min(32767.0, h)));
            }
    }
    T->memFirst.resize(kTileCells * kTileCells + 1);
    std::uint32_t acc = 0;
    for (int c = 0; c < kTileCells * kTileCells; ++c) {
        T->memFirst[c] = acc;
        std::sort(cellm[c].begin(), cellm[c].end());
        for (std::uint32_t m : cellm[c]) T->memList.push_back(m);
        acc += static_cast<std::uint32_t>(cellm[c].size());
    }
    T->memFirst[kTileCells * kTileCells] = acc;
    // overlapping members per cell (open interiors, SAT in the plate frame)
    T->overlap.assign(kTileCells * kTileCells, 0);
    auto local = [&](std::uint32_t mi, D3 *L) {
        const Tri &t = v.tris[members_[mi].tri];
        for (int k = 0; k < 3; ++k) { const D3 d = Sub(V(t.v[k]), o); L[k] = {Dot(d, e1), Dot(d, e2), 0}; }
    };
    auto overlap2 = [](const D3 *A, const D3 *B) {
        const D3 *Tz[2] = {A, B};
        for (int s2 = 0; s2 < 2; ++s2)
            for (int k = 0; k < 3; ++k) {
                const D3 &a = Tz[s2][k], &b = Tz[s2][(k + 1) % 3];
                const double nx = -(b.y - a.y), ny = b.x - a.x;
                double mn[2] = {1e30, 1e30}, mx[2] = {-1e30, -1e30};
                for (int z = 0; z < 2; ++z)
                    for (int i = 0; i < 3; ++i) {
                        const double pp = nx * Tz[z][i].x + ny * Tz[z][i].y;
                        mn[z] = std::min(mn[z], pp); mx[z] = std::max(mx[z], pp);
                    }
                const double l = std::sqrt(nx * nx + ny * ny);
                if (mx[0] <= mn[1] + 1e-6 * l || mx[1] <= mn[0] + 1e-6 * l) return false;
            }
        return true;
    };
    for (int c = 0; c < kTileCells * kTileCells; ++c) {
        const std::uint32_t a0 = T->memFirst[c], a1 = T->memFirst[c + 1];
        for (std::uint32_t a = a0; a < a1 && !T->overlap[c]; ++a)
            for (std::uint32_t b = a + 1; b < a1 && !T->overlap[c]; ++b) {
                D3 LA[3], LB[3];
                local(T->memList[a], LA); local(T->memList[b], LB);
                if (overlap2(LA, LB)) T->overlap[c] = 1;
            }
    }
    return T;
}

void Atlas::ColumnRecords(std::uint32_t ti, std::int32_t ixA, std::int32_t ixB, std::int32_t izA, std::int32_t izB,
                          std::vector<ColRec> *recs) const {
    const Tri &t = view_->tris[ti];
    const std::uint32_t p = plateOf_[ti] == kNoPlate ? kLoose : plateOf_[ti];
    D3 L[3];
    for (int k = 0; k < 3; ++k) L[k] = {double(t.v[k][0]), double(t.v[k][2]), double(t.v[k][1])};
    for (int cls = 0; cls < 2; ++cls) {
        const double R = kColumnRadius[cls];
        const std::int32_t x0 = FloorDiv(double(t.lo[0]) - R - colX0_, kColumnM), x1 = FloorDiv(double(t.hi[0]) + R - colX0_, kColumnM);
        const std::int32_t z0 = FloorDiv(double(t.lo[2]) - R - colZ0_, kColumnM), z1 = FloorDiv(double(t.hi[2]) + R - colZ0_, kColumnM);
        for (std::int32_t iz = std::max(izA, z0); iz <= std::min(izB, z1); ++iz)
            for (std::int32_t ix = std::max(ixA, x0); ix <= std::min(ixB, x1); ++ix) {
                const double cx0 = colX0_ + ix * double(kColumnM) - R, cx1 = colX0_ + (ix + 1) * double(kColumnM) + R;
                const double cz0 = colZ0_ + iz * double(kColumnM) - R, cz1 = colZ0_ + (iz + 1) * double(kColumnM) + R;
                double b[6];
                if (!ClipBounds(L, cx0, cx1, cz0, cz1, b)) continue;
                ColRec r;
                r.key = (std::uint64_t(std::uint32_t(iz) * colNx_ + std::uint32_t(ix)) * 2u + std::uint32_t(cls));
                r.plate = p; r.tri = ti;
                r.y0 = static_cast<std::int32_t>(std::floor(b[4] * 1000.0)) - 1;
                r.y1 = static_cast<std::int32_t>(std::ceil(b[5] * 1000.0)) + 1;
                // the corner as the run time forms it (float); the box steps outward with a 1e-4 m slack
                const float fx0 = colX0_ + float(ix) * kColumnM - kColumnRadius[cls], fz0 = colZ0_ + float(iz) * kColumnM - kColumnRadius[cls];
                const double st = ColumnBoxStep(cls);
                auto q = [&](double x, double c0, bool up) {
                    const double tt = (x - double(c0)) / st;
                    const double vv = up ? std::ceil(tt + 1e-4 / st) : std::floor(tt - 1e-4 / st);
                    return static_cast<std::uint8_t>(std::max(0.0, std::min(255.0, vv)));
                };
                r.box[0] = q(b[0], fx0, false); r.box[1] = q(b[1], fx0, true);
                r.box[2] = q(b[2], fz0, false); r.box[3] = q(b[3], fz0, true);
                recs->push_back(r);
            }
    }
}

bool Atlas::MembersOverlap(std::uint32_t ma, std::uint32_t mb) const {   // two members of one plate overlap in its frame (SAT)
    const world::WorldPackView &v = *view_;
    const PlateRec &P = plates_[members_[ma].plate];
    const D3 o = Vd(P.o), e1 = V(P.e1), e2 = V(P.e2);
    D3 A[3], B[3];
    for (int k = 0; k < 3; ++k) {
        const D3 da = Sub(V(v.tris[members_[ma].tri].v[k]), o), db = Sub(V(v.tris[members_[mb].tri].v[k]), o);
        A[k] = {Dot(da, e1), Dot(da, e2), 0}; B[k] = {Dot(db, e1), Dot(db, e2), 0};
    }
    const D3 *Tz[2] = {A, B};
    for (int s2 = 0; s2 < 2; ++s2)
        for (int k = 0; k < 3; ++k) {
            const D3 &a = Tz[s2][k], &b = Tz[s2][(k + 1) % 3];
            const double nx = -(b.y - a.y), ny = b.x - a.x;
            double mn[2] = {1e30, 1e30}, mx[2] = {-1e30, -1e30};
            for (int z = 0; z < 2; ++z)
                for (int q = 0; q < 3; ++q) {
                    const double pp = nx * Tz[z][q].x + ny * Tz[z][q].y;
                    mn[z] = std::min(mn[z], pp); mx[z] = std::max(mx[z], pp);
                }
            const double l = std::sqrt(nx * nx + ny * ny);
            if (mx[0] <= mn[1] + 1e-6 * l || mx[1] <= mn[0] + 1e-6 * l) return false;
        }
    return true;
}

// recs sorted by (key, plate, tri); slotOf(key) is the output slot in [0, nslots). Appends to the output arrays
// exactly what the run time reads: per slot the entries (sorted by y0, prefix max of y1), the union, and per
// entry the wheel-class member list and overlap flag.
template <class SlotOf>
void Atlas::EmitSlots(const std::vector<ColRec> &recs, SlotOf slotOf, std::size_t nslots, Emitted *o) const {
    o->first.assign(nslots + 1, 0);
    o->unionFirst.assign(nslots + 1, 0);
    o->entries.clear(); o->uni.clear(); o->entMemFirst.clear(); o->entMembers.clear(); o->overlap.clear();
    std::vector<std::vector<std::uint32_t>> emem;
    std::size_t i = 0, next = 0;
    while (i < recs.size()) {
        const std::uint64_t key = recs[i].key;
        const std::size_t slot = slotOf(key);
        while (next <= slot) {
            o->first[next] = static_cast<std::uint32_t>(o->entries.size());
            o->unionFirst[next] = static_cast<std::uint32_t>(o->uni.size() / 2);
            ++next;
        }
        const bool wheelCls = (key & 1u) == 0u;
        const std::size_t first = o->entries.size();
        emem.clear();
        while (i < recs.size() && recs[i].key == key) {
            ColumnEntry e{recs[i].plate, recs[i].y0, recs[i].y1, 0, {recs[i].box[0], recs[i].box[1], recs[i].box[2], recs[i].box[3]}};
            std::vector<std::uint32_t> mm;
            std::size_t j = i;
            while (j < recs.size() && recs[j].key == key && recs[j].plate == e.plate) {
                e.y0 = std::min(e.y0, recs[j].y0); e.y1 = std::max(e.y1, recs[j].y1);
                e.box[0] = std::min(e.box[0], recs[j].box[0]); e.box[1] = std::max(e.box[1], recs[j].box[1]);
                e.box[2] = std::min(e.box[2], recs[j].box[2]); e.box[3] = std::max(e.box[3], recs[j].box[3]);
                if (wheelCls && e.plate != kLoose) mm.push_back(memberOf_[recs[j].tri]);
                ++j;
            }
            o->entries.push_back(e);
            emem.push_back(std::move(mm));
            i = j;
        }
        std::vector<std::size_t> idx(o->entries.size() - first);
        for (std::size_t q = 0; q < idx.size(); ++q) idx[q] = q;
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            const ColumnEntry &A = o->entries[first + a], &B = o->entries[first + b];
            return A.y0 != B.y0 ? A.y0 < B.y0 : A.plate < B.plate;
        });
        std::vector<ColumnEntry> sorted(idx.size());
        for (std::size_t q = 0; q < idx.size(); ++q) sorted[q] = o->entries[first + idx[q]];
        std::int32_t m = INT32_MIN;
        for (std::size_t q = 0; q < sorted.size(); ++q) {
            m = std::max(m, sorted[q].y1); sorted[q].y1max = m;
            o->entries[first + q] = sorted[q];
            std::vector<std::uint32_t> &mm = emem[idx[q]];
            std::sort(mm.begin(), mm.end());
            mm.erase(std::unique(mm.begin(), mm.end()), mm.end());
            o->entMemFirst.push_back(static_cast<std::uint32_t>(o->entMembers.size()));
            bool ov = mm.size() > 64;   // too many to check: treat as overlapping (the disk proof is then not used)
            for (std::size_t a = 0; a < mm.size() && !ov; ++a)
                for (std::size_t b = a + 1; b < mm.size() && !ov; ++b) ov = MembersOverlap(mm[a], mm[b]);
            o->overlap.push_back(ov ? 1 : 0);
            for (std::uint32_t x : mm) o->entMembers.push_back(x);
        }
        std::int32_t u0 = 0, u1 = 0;
        bool open = false;
        for (std::size_t q = 0; q < sorted.size(); ++q) {
            if (!open) { u0 = sorted[q].y0; u1 = sorted[q].y1; open = true; continue; }
            if (sorted[q].y0 <= u1) u1 = std::max(u1, sorted[q].y1);
            else { o->uni.push_back(u0); o->uni.push_back(u1); u0 = sorted[q].y0; u1 = sorted[q].y1; }
        }
        if (open) { o->uni.push_back(u0); o->uni.push_back(u1); }
    }
    while (next <= nslots) {
        o->first[next] = static_cast<std::uint32_t>(o->entries.size());
        o->unionFirst[next] = static_cast<std::uint32_t>(o->uni.size() / 2);
        ++next;
    }
    o->entMemFirst.push_back(static_cast<std::uint32_t>(o->entMembers.size()));
}

void Atlas::ColumnFrame() {
    const world::WorldPackView &v = *view_;
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (const Tri &t : v.tris)
        for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], t.lo[k]); hi[k] = std::max(hi[k], t.hi[k]); }
    if (v.tris.empty()) return;
    const float pad = kColumnRadius[1] + kColumnM;
    colX0_ = std::floor((lo[0] - pad) / kColumnM) * kColumnM;
    colZ0_ = std::floor((lo[2] - pad) / kColumnM) * kColumnM;
    colNx_ = static_cast<std::uint32_t>(std::ceil((hi[0] + pad - colX0_) / kColumnM));
    colNz_ = static_cast<std::uint32_t>(std::ceil((hi[2] + pad - colZ0_) / kColumnM));
    worldY0_ = lo[1] - 1.0f; worldY1_ = hi[1] + 1.0f;
    stats.columns = std::size_t(colNx_) * colNz_;
}

void Atlas::BuildColumns() {
    const world::WorldPackView &v = *view_;
    ColumnFrame();
    if (v.tris.empty()) return;
    const std::size_t ncol = std::size_t(colNx_) * colNz_;
    std::vector<ColRec> recs;
    recs.reserve(v.tris.size() * 12);
    for (std::size_t ti = 0; ti < v.tris.size(); ++ti)
        ColumnRecords(static_cast<std::uint32_t>(ti), 0, std::int32_t(colNx_) - 1, 0, std::int32_t(colNz_) - 1, &recs);
    std::sort(recs.begin(), recs.end(), [](const ColRec &a, const ColRec &b) {
        return a.key != b.key ? a.key < b.key : a.plate != b.plate ? a.plate < b.plate : a.tri < b.tri;
    });
    Emitted all;
    EmitSlots(recs, [](std::uint64_t key) { return static_cast<std::size_t>(key); }, ncol * 2, &all);
    colFirst_.swap(all.first);
    colUnionFirst_.swap(all.unionFirst);
    colEntries_.swap(all.entries);
    colUnion_.swap(all.uni);
    entMemFirst_.swap(all.entMemFirst);
    entMembers_.swap(all.entMembers);
    entOverlap_.swap(all.overlap);
    colEntries_.shrink_to_fit();
    colUnion_.shrink_to_fit();
    entMembers_.shrink_to_fit();
    stats.columnUnion = colUnion_.size() / 2;
    stats.entryMembers = entMembers_.size();
    stats.columnEntries = colEntries_.size();
}

bool Atlas::BuildColumnBlock(std::uint32_t bx, std::uint32_t bz, ColumnBlock *out) const {
    const std::int32_t ixA = std::int32_t(bx * kColumnBlock), izA = std::int32_t(bz * kColumnBlock);
    if (ixA >= std::int32_t(colNx_) || izA >= std::int32_t(colNz_)) return false;
    const std::int32_t ixB = std::min<std::int32_t>(ixA + kColumnBlock, std::int32_t(colNx_)) - 1;
    const std::int32_t izB = std::min<std::int32_t>(izA + kColumnBlock, std::int32_t(colNz_)) - 1;
    // every triangle whose AABB, dilated by the body class radius, meets the block's columns
    const float R = kColumnRadius[1] + 0.01f;
    Box3 q;
    q.lo[0] = colX0_ + float(ixA) * kColumnM - R; q.hi[0] = colX0_ + float(ixB + 1) * kColumnM + R;
    q.lo[2] = colZ0_ + float(izA) * kColumnM - R; q.hi[2] = colZ0_ + float(izB + 1) * kColumnM + R;
    q.lo[1] = worldY0_; q.hi[1] = worldY1_;
    std::vector<ColRec> recs;
    bvh_->Query(q, [&](std::uint32_t ti) { ColumnRecords(ti, ixA, ixB, izA, izB, &recs); });
    std::sort(recs.begin(), recs.end(), [](const ColRec &a, const ColRec &b) {
        return a.key != b.key ? a.key < b.key : a.plate != b.plate ? a.plate < b.plate : a.tri < b.tri;
    });
    // local slot = ((iz - izA) * kColumnBlock + (ix - ixA)) * 2 + cls
    const std::uint32_t nx = colNx_;
    Emitted e;
    constexpr std::size_t kSlots = std::size_t(kColumnBlock) * kColumnBlock * 2u;
    EmitSlots(recs, [&](std::uint64_t key) {
        const std::uint64_t col = key >> 1, cls = key & 1u;
        const std::int32_t ix = std::int32_t(col % nx), iz = std::int32_t(col / nx);
        return static_cast<std::size_t>((std::size_t(iz - izA) * kColumnBlock + std::size_t(ix - ixA)) * 2u + cls);
    }, kSlots, &e);
    out->bx = bx; out->bz = bz;
    for (std::size_t q = 0; q <= kSlots; ++q) { out->first[q] = e.first[q]; out->unionFirst[q] = e.unionFirst[q]; }
    out->entries.swap(e.entries);
    out->unions.swap(e.uni);
    out->memFirst.swap(e.entMemFirst);
    out->members.swap(e.entMembers);
    out->overlap.swap(e.overlap);
    return true;
}

}  // namespace tw::atlas
