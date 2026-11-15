#include "atlas_stage.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

namespace tw::atlas {

const char AtlasCache::kKind = 0;

AtlasCache::AtlasCache() : step::CollisionCache(&kKind) {}

void AtlasCache::Flush() {
    exact.Flush();
    for (auto &row : hint)
        for (auto &h : row) h = TileHint{};
    owner = 0;
}

namespace {

std::atomic<std::uint64_t> g_nextId{1};

inline std::int32_t FloorI(float x) {
    const std::int32_t i = static_cast<std::int32_t>(x);
    return (static_cast<float>(i) > x) ? i - 1 : i;
}


}  // namespace

bool AtlasCollision::Init(const world::WorldPackView &view, const collide::CarShapes &car, std::string *error, const AtlasOptions &opt) {
    opt_ = opt;
    view_ = &view;
    if (!exact_.Init(view, car, error)) return false;
    if (!atlas_.Build(view, exact_.Bvh(), opt.build, error)) return false;
    if (!fb_.Init(exact_, error)) return false;
    atlas_.ColumnGrid(&colX0_, &colZ0_, &colNx_, &colNz_);
    if (opt.build.lazyColumns) {
        ColumnCache::Options co;
        co.budgetBytes = opt.columnBudget;
        const Atlas *a = &atlas_;
        colCache_.reset(new ColumnCache(colX0_, colZ0_, colNx_, colNz_,
                                        [a](std::uint32_t bx, std::uint32_t bz, ColumnBlock *out) { return a->BuildColumnBlock(bx, bz, out); }, co));
    }
    id_ = g_nextId.fetch_add(1);
    return true;
}

std::unique_ptr<step::CollisionCache> AtlasCollision::NewCache() const { return std::make_unique<AtlasCache>(); }

void AtlasCollision::Detect(const model::TwinState &s, step::SubstepContacts &out) const { DetectPose(s.pos, s.quat, s.absorb, out); }

void AtlasCollision::DetectCached(const model::TwinState &s, step::CollisionCache *cache, step::SubstepContacts &out) const {
    DetectPose(s.pos, s.quat, s.absorb, out, cache);
}

void AtlasCollision::DetectPose(const float pos[3], const float quat[4], const float absorb[4], step::SubstepContacts &out,
                                step::CollisionCache *cache) const {
    collide::ShapePose sp[8];
    collide::ShapePoses(exact_.Car(), pos, quat, absorb, sp);
    AtlasCache *c = cache != nullptr && cache->kind == &AtlasCache::kKind ? static_cast<AtlasCache *>(cache) : nullptr;
    if (c == nullptr) {
        thread_local AtlasCache local;
        c = &local;
    }
    if (c->flushEveryCall) c->Flush();
    DetectShapes(sp, out, c);
}

const Tile *AtlasCollision::Lookup(AtlasCache &cache, int k, int slot, std::uint32_t plate, std::int32_t tx, std::int32_t ty) const {
    AtlasCache::TileHint &h = cache.hint[k][slot];
    if (cache.owner == id_ && h.plate == plate && h.tx == tx && h.ty == ty) return h.t;
    if (cache.owner != id_) {
        for (auto &row : cache.hint)
            for (auto &x : row) x = AtlasCache::TileHint{};
        cache.owner = id_;
    }
    const Tile *t = atlas_.TileAt(plate, tx, ty);
    h.plate = plate; h.tx = tx; h.ty = ty; h.t = t;
    return t;
}

int AtlasCollision::Reader() const {
    struct Reg { const ColumnCache *c; int r; };
    thread_local Reg regs[4] = {{nullptr, -1}, {nullptr, -1}, {nullptr, -1}, {nullptr, -1}};
    for (Reg &g : regs) if (g.c == colCache_.get()) return g.r;
    for (Reg &g : regs)
        if (g.c == nullptr) { g.c = colCache_.get(); g.r = colCache_->Register(); return g.r; }
    regs[0].c = colCache_.get();   // more than 4 stages on one thread: re-register (rare)
    regs[0].r = colCache_->Register();
    return regs[0].r;
}

bool AtlasCollision::Column(const float x[3], int cls, ColView *v) const {
    const float fx = (x[0] - colX0_) * (1.0f / kColumnM), fz = (x[2] - colZ0_) * (1.0f / kColumnM);
    if (!(fx >= 0.0f && fz >= 0.0f)) return false;
    const std::uint32_t ix = static_cast<std::uint32_t>(fx), iz = static_cast<std::uint32_t>(fz);
    if (ix >= colNx_ || iz >= colNz_) return false;
    v->cx0 = colX0_ + float(ix) * kColumnM - kColumnRadius[cls];   // = Atlas::SlotCorner (the build's float formula)
    v->cz0 = colZ0_ + float(iz) * kColumnM - kColumnRadius[cls];
    if (colCache_) {
        const int r = Reader();
        if (r < 0) return false;
        const std::uint32_t h = colCache_->SlotHandle(x, cls);
        if (h == 0xFFFFFFFFu) return false;
        v->ref = colCache_->Resolve(r, h);
        if (v->ref.b == nullptr) return false;
        v->e = v->ref.entries; v->n = v->ref.n; v->u = v->ref.unions; v->nu = v->ref.unionCount;
        return true;
    }
    const std::uint32_t slot = (iz * colNx_ + ix) * 2u + static_cast<std::uint32_t>(cls);
    v->e = atlas_.Entries(slot, &v->n);
    v->u = atlas_.Union(slot, &v->nu);
    v->globalFirst = atlas_.EntryIndex(v->e);
    return true;
}

const std::uint32_t *AtlasCollision::ColumnMembers(const ColView &v, std::uint32_t i, std::uint32_t *count, bool *overlap) const {
    if (colCache_) return ColumnCache::EntryMembers(v.ref, i, count, overlap);
    return atlas_.EntryMembers(v.globalFirst + i, count, overlap);
}

AtlasCollision::TryResult AtlasCollision::TryPlate(const collide::ShapePose &sp, int k, int ci, const std::uint32_t *cand,
                                                   const float *candZ, const float *candH, int ncand, model::RespContact *c,
                                                   AtlasCache &cache, AtlasStats *stats, FallbackWhy *why) const {
    const std::uint32_t plate = cand[ci];
    const PlateRec &P = atlas_.Plates()[plate];
    const float *R = sp.world.r;   // columns: axis_j = R[3j .. 3j+2]
    const float *r = sp.radii;
    // centre in the plate frame (the subtraction in double: plate origins are world coordinates)
    const double d0[3] = {double(sp.world.t[0]) - P.o[0], double(sp.world.t[1]) - P.o[1], double(sp.world.t[2]) - P.o[2]};
    const float u = float(d0[0] * P.e1[0] + d0[1] * P.e1[1] + d0[2] * P.e1[2]);
    const float v = float(d0[0] * P.e2[0] + d0[1] * P.e2[1] + d0[2] * P.e2[2]);
    const float z = float(d0[0] * P.n[0] + d0[1] * P.n[1] + d0[2] * P.n[2]);
    float aj[3], c1[3], c2[3];
    float hS2 = 0.0f, a11 = 0.0f, a12 = 0.0f, a22 = 0.0f;
    for (int j = 0; j < 3; ++j) {
        const float *ax = R + 3 * j;
        aj[j] = (ax[0] * P.n[0] + ax[1] * P.n[1] + ax[2] * P.n[2]) * r[j];
        c1[j] = (ax[0] * P.e1[0] + ax[1] * P.e1[1] + ax[2] * P.e1[2]) * r[j];
        c2[j] = (ax[0] * P.e2[0] + ax[1] * P.e2[1] + ax[2] * P.e2[2]) * r[j];
        hS2 += aj[j] * aj[j];
        a11 += c1[j] * c1[j]; a12 += c1[j] * c2[j]; a22 += c2[j] * c2[j];
    }
    const float hS = std::sqrt(hS2);
    const float slo = z - hS, shi = z + hS;
    // ---- foreign test: (1) no other plate of the column overlaps the shape's y-range: clear (exact: the column lists
    // every plate within the class radius); (2) plane tests against the others; (3) the tile field, only when a plane
    // test fails (cover points: wheels 1 at the centre; bodies 1 or 3 along the shadow's major axis)
    bool planesOk = true;
    for (int i = 0; i < ncand && planesOk; ++i) {   // z and the support along n come from the plate order (same bits)
        if (i == ci) continue;
        if (!opt_.planeTests) { planesOk = false; break; }
        if (stats) ++stats->planeTestsRun;
        const float slackQ = atlas_.Plates()[cand[i]].slack;
        planesOk = candZ[i] - candH[i] > slackQ || candZ[i] < -slackQ;
    }
    if (!planesOk) {
        if (stats) ++stats->fieldTests;
        bool fieldOk = z >= kCentreLo && z <= kCentreHi && slo >= kFieldLo + 0.1f && shi <= kFieldHi - 0.1f;
        if (fieldOk) {
            float pu[3] = {u, 0, 0}, pv[3] = {v, 0, 0};
            int np = 1, cls = 1;
            if (k < 4) {
                const float tr = a11 + a22, det = a11 * a22 - a12 * a12;
                const float disc = std::sqrt(std::fmax(0.0f, tr * tr * 0.25f - det));
                const float lam1 = tr * 0.5f + disc, lam2 = std::fmax(0.0f, tr * 0.5f - disc);
                const float l1 = std::sqrt(lam1);
                float dx = lam1 - a22, dy = a12;
                if (dy * dy < 1e-20f) { dx = a11 >= a22 ? 1.0f : 0.0f; dy = a11 >= a22 ? 0.0f : 1.0f; }
                const float il = 1.0f / std::sqrt(dx * dx + dy * dy);
                dx *= il; dy *= il;
                const float r1 = std::sqrt(lam1 + lam2) + 1e-4f, r3 = std::sqrt(lam1 * (1.0f / 9.0f) + lam2) + 1e-4f;
                float rr = r1;
                if (r3 < r1) {
                    rr = r3; np = 3;
                    const float off = (2.0f / 3.0f) * l1;
                    pu[0] = u - off * dx; pv[0] = v - off * dy; pu[1] = u; pv[1] = v; pu[2] = u + off * dx; pv[2] = v + off * dy;
                }
                cls = 0;
                while (cls < kClasses && kClassR[cls] < rr) ++cls;
                if (cls >= kClasses) fieldOk = false;
            } else {
                const float rmax = std::fmax(r[0], std::fmax(r[1], r[2]));
                cls = 0;
                while (cls < kClasses && kClassR[cls] < rmax + 1e-4f) ++cls;
                if (cls >= kClasses) fieldOk = false;
            }
            const std::int32_t ilo = static_cast<std::int32_t>(std::floor(slo * 1000.0f)) - 1;
            const std::int32_t ihi = static_cast<std::int32_t>(std::ceil(shi * 1000.0f)) + 1;
            for (int i = 0; i < np && fieldOk; ++i) {
                const std::int32_t tx = FloorI(pu[i] * (1.0f / kTileM)), ty = FloorI(pv[i] * (1.0f / kTileM));
                const Tile *T = Lookup(cache, k, i, plate, tx, ty);
                if (T == nullptr) { fieldOk = false; break; }
                int ci = static_cast<int>((pu[i] - float(tx) * kTileM) * (1.0f / kCellM));
                int cj = static_cast<int>((pv[i] - float(ty) * kTileM) * (1.0f / kCellM));
                ci = ci < 0 ? 0 : ci > kTileCells - 1 ? kTileCells - 1 : ci;
                cj = cj < 0 ? 0 : cj > kTileCells - 1 ? kTileCells - 1 : cj;
                const CellRec &cr = T->cell[cj * kTileCells + ci];
                fieldOk = ilo > cr.lo[cls] && ihi < cr.hi[cls];
            }
        }
        if (!fieldOk) { *why = kFbForeign; return kTryFail; }
    }
    if (opt_.profile == 4) return kTryFail;
    // ---- the home plate itself
    if (slo > P.slack || z < -P.slack) return kTryT0;
    if (slo > -P.slack) { *why = kFbReachTie; return kTryFail; }
    // foot point on P's plane, its member
    float qe1 = 0.0f, qe2 = 0.0f;   // (M M^T n) . e1, . e2 = sum_j c_j a_j
    for (int j = 0; j < 3; ++j) { qe1 += c1[j] * aj[j]; qe2 += c2[j] * aj[j]; }
    const float kz = z / hS2;
    const float up = u - kz * qe1, vp = v - kz * qe2;
    // the foot point's wheel-class column lists the plate's members within 0.40 m of the column: the containing one
    float qw[3] = {0.0f, 0.0f, 0.0f};   // M M^T n (world)
    for (int j = 0; j < 3; ++j) {
        const float *ax = R + 3 * j;
        for (int e = 0; e < 3; ++e) qw[e] += ax[e] * (aj[j] * r[j]);
    }
    const float foot[3] = {sp.world.t[0] - kz * qw[0], sp.world.t[1] - kz * qw[1], sp.world.t[2] - kz * qw[2]};
    ColView fv;
    if (!Column(foot, 0, &fv)) { *why = kFbMember; return kTryFail; }
    std::uint32_t pi = 0xFFFFFFFFu;
    for (std::uint32_t i = 0; i < fv.n; ++i) if (fv.e[i].plate == plate) { pi = i; break; }
    if (pi == 0xFFFFFFFFu) { *why = kFbMember; return kTryFail; }
    std::uint32_t nm = 0;
    bool overlap = false;
    const std::uint32_t *ml = ColumnMembers(fv, pi, &nm, &overlap);
    const std::vector<MemberRec> &mem = atlas_.Members();
    std::uint32_t found = kNoPlate;
    int nf = 0;
    float emin = 0.0f;
    for (std::uint32_t q = 0; q < nm; ++q) {
        const MemberRec &M = mem[ml[q]];
        const float e0 = M.ln[0][0] * up + M.ln[0][1] * vp + M.ln[0][2];
        const float e1 = M.ln[1][0] * up + M.ln[1][1] * vp + M.ln[1][2];
        const float e2 = M.ln[2][0] * up + M.ln[2][1] * vp + M.ln[2][2];
        const float mn = std::fmin(e0, std::fmin(e1, e2));
        if (mn > -2e-4f) {
            if (mn < 2e-4f) { *why = kFbMember; return kTryFail; }   // within the containment margin of an edge
            ++nf; found = ml[q]; emin = mn;
        }
    }
    if (nf != 1) { *why = kFbMember; return kTryFail; }
    if (opt_.profile == 5) return kTryFail;
    // the FACE, by D's closed form (the same bits in every tier)
    const FaceTri &F = fb_.Faces()[mem[found].tri];
    ShapeLocal sl;
    MakeShapeLocal(sp.world.r, sp.world.t, sp.radii, &sl);
    FaceOut fo;
    float tie = 0.0f;
    if (!ClosedFace(F, sl, &fo, &tie)) { *why = kFbFaceTie; return kTryFail; }
    if (opt_.profile == 6) return kTryFail;
    // merge proof: no other member reaches the contact disk, or FV's 30-degree rule drops every coplanar feature
    const float d = fo.d;
    const float hh = std::sqrt(std::fmax(0.0f, 1.0f - d * d));
    const float rmax = std::fmax(r[0], std::fmax(r[1], r[2])), rmin = std::fmin(r[0], std::fmin(r[1], r[2]));
    bool proven = !overlap && emin > hh * rmax + 3e-4f;
    if (!proven) {
        // atan(hh sig / den) + 2 alpha + 2e-3 < 30 deg, as a comparison with the plate's precomputed tangent (no libm)
        const float sig = 1.0f / rmin, den = d / hS - hh * sig;
        proven = den > 0.0f && hh * sig < P.tanProof * den;
    }
    if (!proven) { *why = kFbProof; return kTryFail; }
    const MemberRec &M = mem[found];
    const Tri &tr = view_->tris[M.tri];
    const world::TriInfo &ti = view_->triInfo[M.tri];
    const world::Actor &a = view_->ActorOf(tr);
    FaceToWorld(fo, c);
    c->peerZ[0] = a.peerZ[0]; c->peerZ[1] = a.peerZ[1]; c->peerZ[2] = a.peerZ[2];
    const bool trig = ti.trigger != world::kNoTrigger;   // never on a plate (triggers are loose); kept for symmetry
    c->peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
    c->shape = static_cast<std::uint8_t>(k);
    c->material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
    c->flags = static_cast<std::uint8_t>(model::kRespPrimary | (trig ? unsigned(model::kRespTrigger) : 0u));
    c->pad = 0u;
    return kTryT1;
}

Decision AtlasCollision::Decide(const collide::ShapePose &sp, int k, model::RespContact *c, AtlasCache &cache, AtlasStats *stats,
                                FallbackWhy *why) const {
    const float rmax = std::fmax(sp.radii[0], std::fmax(sp.radii[1], sp.radii[2]));
    const int ccls = rmax <= kColumnRadius[0] ? 0 : 1;
    if (rmax > kColumnRadius[1]) { *why = kFbForeign; return kFallback; }
    ColView cv;
    if (!Column(sp.world.t, ccls, &cv)) return kFree;   // outside the grid: no geometry within reach
    const std::int32_t ylo = static_cast<std::int32_t>(std::floor((sp.world.t[1] - rmax) * 1000.0f)) - 1;
    const std::int32_t yhi = static_cast<std::int32_t>(std::ceil((sp.world.t[1] + rmax) * 1000.0f)) + 1;
    // the shape's x-z box in the slot's u8 steps (outward)
    const float sx0 = cv.cx0, sz0 = cv.cz0;
    const float ist = 1.0f / ColumnBoxStep(ccls);
    const float bx0 = (sp.world.t[0] - rmax - sx0) * ist - 1.0f, bx1 = (sp.world.t[0] + rmax - sx0) * ist + 1.0f;
    const float bz0 = (sp.world.t[2] - rmax - sz0) * ist - 1.0f, bz1 = (sp.world.t[2] + rmax - sz0) * ist + 1.0f;
    {
        bool hitU = false;
        for (std::uint32_t i = 0; i < cv.nu && !hitU; ++i) hitU = cv.u[2 * i] <= yhi && cv.u[2 * i + 1] >= ylo;
        if (!hitU) return kFree;
        if (opt_.profile == 1) return kFallback;
    }
    constexpr int kMaxCand = 16;
    std::uint32_t cand[kMaxCand];
    int n = 0;
    // first entry whose prefix-max y1 reaches ylo (binary search), then scan until y0 > yhi
    std::uint32_t lo = 0, hi = cv.n;
    while (lo < hi) {
        const std::uint32_t mid = (lo + hi) >> 1;
        if (cv.e[mid].y1max < ylo) lo = mid + 1; else hi = mid;
    }
    if (stats) stats->colScanned += cv.n - lo;
    for (std::uint32_t i = lo; i < cv.n; ++i) {
        const ColumnEntry &e = cv.e[i];
        if (e.y0 > yhi) break;
        if (e.y1 < ylo) continue;
        if (float(e.box[1]) < bx0 || float(e.box[0]) > bx1 || float(e.box[3]) < bz0 || float(e.box[2]) > bz1) continue;
        if (e.plate == kLoose) { *why = kFbLoose; return kFallback; }
        if (n == kMaxCand) { *why = kFbForeign; return kFallback; }
        cand[n++] = e.plate;
    }
    if (n == 0) return kFree;
    if (opt_.profile == 2) return kFallback;
    // candidate order: the gap score (most penetrated plane in front first), ties by plate id: a function of the pose.
    // z and the support along n are kept for the plane tests and the home plate (the same bits as TryPlate's own form)
    float score[kMaxCand], candZ[kMaxCand], candH[kMaxCand];
    for (int i = 0; i < n; ++i) {
        const PlateRec &P = atlas_.Plates()[cand[i]];
        const double d0[3] = {double(sp.world.t[0]) - P.o[0], double(sp.world.t[1]) - P.o[1], double(sp.world.t[2]) - P.o[2]};
        const float z = float(d0[0] * P.n[0] + d0[1] * P.n[1] + d0[2] * P.n[2]);
        float h2 = 0.0f;
        for (int j = 0; j < 3; ++j) {
            const float *ax = sp.world.r + 3 * j;
            const float a = (ax[0] * P.n[0] + ax[1] * P.n[1] + ax[2] * P.n[2]) * sp.radii[j];
            h2 += a * a;
        }
        candZ[i] = z;
        candH[i] = std::sqrt(h2);
        score[i] = z >= -0.05f ? z - candH[i] : 100.0f - z;
    }
    int order[kMaxCand];
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order, order + n, [&](int a, int b) { return score[a] < score[b] || (score[a] == score[b] && cand[a] < cand[b]); });
    if (opt_.profile == 3) return kFallback;
    const int tries = std::min(n, opt_.maxTries);
    FallbackWhy w = kFbTries;
    for (int t = 0; t < tries; ++t) {
        if (stats) ++stats->tries;
        FallbackWhy wt = kFbTries;
        const TryResult r = TryPlate(sp, k, order[t], cand, candZ, candH, n, c, cache, stats, &wt);
        if (r == kTryT0) return kT0;
        if (r == kTryT1) return kT1;
        if (t == 0) w = wt;
    }
    *why = w;
    return kFallback;
}

bool AtlasCollision::TickClear(const model::TwinState &s, float dt, float aMax) const {
    collide::ShapePose sp[8];
    collide::ShapePoses(exact_.Car(), s.pos, s.quat, s.absorb, sp);
    return TickClearShapes(sp, s.pos, s.vel, s.angVel, dt, aMax);
}

bool AtlasCollision::TickClearShapes(const collide::ShapePose sp[8], const float pos[3], const float vel[3], const float angVel[3], float dt,
                                     float aMax) const {
    struct { const float *pos; } s{pos};
    const float vm = std::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
    const float wm = std::sqrt(angVel[0] * angVel[0] + angVel[1] * angVel[1] + angVel[2] * angVel[2]);
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    float rcar = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const float rmax = std::fmax(sp[k].radii[0], std::fmax(sp[k].radii[1], sp[k].radii[2]));
        float d2 = 0.0f;
        for (int a = 0; a < 3; ++a) { const float d = sp[k].world.t[a] - s.pos[a]; d2 += d * d; }
        rcar = std::fmax(rcar, std::sqrt(d2) + rmax);
    }
    // the car turns about its COM (|COM| <= 0.6 m from the car origin for StadiumCar: +0.6 on the lever), and |w| can
    // grow inside a tick (C14's pins restore the remembered yaw: x1.021 measured; x1.05 + 0.1, capped at the 100 rad/s clamp)
    const float wb = std::fmin(100.0f, wm * 1.05f + 0.1f);
    const float travel = vm * dt + 0.5f * aMax * dt * dt + wb * dt * (rcar + 0.6f) + 0.25f;
    for (int k = 0; k < 8; ++k) {
        const float rmax = std::fmax(sp[k].radii[0], std::fmax(sp[k].radii[1], sp[k].radii[2]));
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::fmin(lo[a], sp[k].world.t[a] - rmax - travel);
            hi[a] = std::fmax(hi[a], sp[k].world.t[a] + rmax + travel);
        }
    }
    const std::int32_t ylo = static_cast<std::int32_t>(std::floor(lo[1] * 1000.0f)) - 1;
    const std::int32_t yhi = static_cast<std::int32_t>(std::ceil(hi[1] * 1000.0f)) + 1;
    // every column whose rectangle meets the swept x-z box (wheel class: its lists hold geometry within 0.40 m of it)
    const float fx0 = (lo[0] - colX0_) * (1.0f / kColumnM), fx1 = (hi[0] - colX0_) * (1.0f / kColumnM);
    const float fz0 = (lo[2] - colZ0_) * (1.0f / kColumnM), fz1 = (hi[2] - colZ0_) * (1.0f / kColumnM);
    if (fx1 < 0.0f || fz1 < 0.0f || fx0 >= float(colNx_) || fz0 >= float(colNz_)) return true;   // outside the grid
    const std::int32_t ix0 = std::max(0, FloorI(fx0)), ix1 = std::min<std::int32_t>(std::int32_t(colNx_) - 1, FloorI(fx1));
    const std::int32_t iz0 = std::max(0, FloorI(fz0)), iz1 = std::min<std::int32_t>(std::int32_t(colNz_) - 1, FloorI(fz1));
    if ((ix1 - ix0 + 1) * (iz1 - iz0 + 1) > 256) return false;   // a very fast tick: no proof
    for (std::int32_t iz = iz0; iz <= iz1; ++iz)
        for (std::int32_t ix = ix0; ix <= ix1; ++ix) {
            const float c[3] = {colX0_ + (float(ix) + 0.5f) * kColumnM, 0.0f, colZ0_ + (float(iz) + 0.5f) * kColumnM};
            ColView v;
            if (!Column(c, 0, &v)) continue;
            for (std::uint32_t i = 0; i < v.nu; ++i)
                if (v.u[2 * i] <= yhi && v.u[2 * i + 1] >= ylo) return false;
        }
    return true;
}

void AtlasCollision::DetectShapes(const collide::ShapePose shapes[8], step::SubstepContacts &out, AtlasCache *cache, AtlasStats *stats,
                                  Decision *decisions) const {
    if (cache == nullptr) {
        thread_local AtlasCache local;
        cache = &local;
    }
    if (colCache_) { const int r = Reader(); if (r >= 0) colCache_->BeginTick(r); }   // no column pointer outlives a call
    model::RespContact face[8];
    Decision dec[8];
    bool anyFallback = false;
    for (int k = 0; k < 8; ++k) {
        FallbackWhy why = kFbTries;
        dec[k] = Decide(shapes[k], k, &face[k], *cache, stats, &why);
        if (dec[k] == kFallback) {
            anyFallback = true;
            if (stats) ++stats->why[why];
        }
        if (stats) ++stats->shapes[dec[k]];
        if (decisions) decisions[k] = dec[k];
    }
    if (stats) ++stats->calls;
    out.n = 0;
    out.overflow = false;
    if (!anyFallback) {
        for (int k = 0; k < 8; ++k)
            if (dec[k] == kT1) out.c[out.n++] = face[k];
        return;
    }
    if (stats) ++stats->fallbackCalls;
    if (opt_.exactFallback) {   // test: B's exact stage for the substep, its contacts of the fallback shapes
        thread_local step::SubstepContacts ex;
        exact_.DetectShapes(shapes, ex, nullptr, &cache->exact);
        for (int k = 0; k < 8; ++k) {
            if (dec[k] == kT1) {
                if (out.n < model::kRespMaxContacts) out.c[out.n++] = face[k]; else out.overflow = true;
            } else if (dec[k] == kFallback) {
                for (int q = 0; q < ex.n; ++q)
                    if (ex.c[q].shape == k) { if (out.n < model::kRespMaxContacts) out.c[out.n++] = ex.c[q]; else out.overflow = true; }
            }
        }
        if (ex.overflow) out.overflow = true;
        return;
    }
    // D's per-shape fallback on the shape's BVH candidates, in shape order (the output stays shape-major)
    thread_local std::vector<std::uint32_t> cand;
    const double origin[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < 8; ++k) {
        if (dec[k] == kT1) {
            if (out.n < model::kRespMaxContacts) out.c[out.n++] = face[k]; else out.overflow = true;
        } else if (dec[k] == kFallback) {
            cand.clear();
            fb_.BvhCandidates(shapes[k], &cand);
            fb_.DetectShape(shapes, k, cand.data(), cand.size(), origin, out);
        }
    }
}

}  // namespace tw::atlas
