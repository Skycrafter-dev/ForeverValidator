// M3 collision for batch lanes: see tw_collide_lanes.h.
#include "tw_collide_lanes.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <x86intrin.h>

namespace tw {
namespace collide {

namespace {

// geometry_helpers.cpp NearlyEqual (the stage's merge)
inline bool NearlyEqual(float value, float reference) {
    const float tol = std::fabs(reference) * 1.0e-5f;
    return reference - tol <= value && value <= reference + tol;
}
constexpr float kNormalAlignment = 0.8660254f;   // PhysicsTolerance::SphereContactNormalAlignment

// The stage's query box of a shape (DetectShapes: closed, inflated for rounding), written in place (a
// returned Box3 copied as 8-byte words over 4-byte stores stalls store forwarding).
inline void ShapeBox(const ShapePose &sp, Box3 *q) {
    for (int a = 0; a < 3; ++a) {
        const float c = sp.world.t[a];
        const float h = std::fabs(sp.world.r[a]) * sp.radii[0] + std::fabs(sp.world.r[3 + a]) * sp.radii[1] + std::fabs(sp.world.r[6 + a]) * sp.radii[2];
        const float eps = 1e-3f + 1e-6f * std::fabs(c);
        q->lo[a] = c - h - eps;
        q->hi[a] = c + h + eps;
    }
}
inline void Grow(Box3 *u, const Box3 &b) {
    for (int a = 0; a < 3; ++a) {
        u->lo[a] = u->lo[a] < b.lo[a] ? u->lo[a] : b.lo[a];
        u->hi[a] = u->hi[a] > b.hi[a] ? u->hi[a] : b.hi[a];
    }
}

}  // namespace

bool ExactCollisionLanes::Init(const world::WorldPackView &view, const ExactCollision &single, std::string *error) {
    view_ = &view;
    single_ = &single;
    // the stage's BVH, FastTri and model bounds (read-only); the TriVerts table is this stage's
    verts_.resize(view.modelTris.size());
    for (std::size_t mi = 0; mi < view.models.size(); ++mi) {
        const world::Model &m = view.models[mi];
        for (std::uint32_t t = 0; t < m.triCount; ++t) {
            const world::ModelTri &mt = view.modelTris[m.firstTri + t];
            exact::TriVerts &tv = verts_[m.firstTri + t];
            for (int c = 0; c < 3; ++c) {
                const world::Vec3f &p = view.modelVerts[m.firstVert + mt.v[c]];
                tv.v[c] = exact::V{p.x, p.y, p.z};
            }
        }
    }
    const CarShapes &car = single.Car();
    for (int k = 0; k < 8; ++k) {
        std::memcpy(scRadii_[k], car.shape[k].radii, sizeof scRadii_[k]);
        sc_[k] = cq::MakeShapeConst(car.shape[k].radii);
    }
    for (PoolMemo &p : pool_) p.have = false;
    for (PoolMemo &p : own_) p.have = false;
    if (error) error->clear();
    return true;
}

void ExactCollisionLanes::DetectLanes(int L, const model::TwinState *const *poses, std::uint32_t mask, step::CollisionCache *const *caches,
                                      step::SubstepContacts *out) const {
    if (sharedMemo_) {
        DetectLanesShared(L, poses, mask, caches, out);
        return;
    }
    ++stats_.calls;
    // Lanes at the same pose bits get the same contacts: search candidates from one snapshot share their
    // trajectory up to their first change. Each distinct pose goes through the single-run stage with its
    // lane's own cache (DetectCached: flush mode, last answer, memo); a copy lane gets that answer and,
    // as DetectPose would leave it, its own flush and last answer.
    std::uint32_t key[kMaxLanes][11];
    std::uint64_t hash[kMaxLanes];
    int dupOf[kMaxLanes];
    std::uint32_t uniq = 0u;
    const std::uint64_t id = single_->Id();
    for (int l = 0; l < L; ++l) {
        if (!((mask >> l) & 1u)) continue;
        const model::TwinState &s = *poses[l];
        std::uint32_t *k = key[l];
        std::memcpy(k, s.pos, 12);
        std::memcpy(k + 3, s.quat, 16);
        std::memcpy(k + 7, s.absorb, 16);
        std::uint64_t h = 1469598103934665603ull;
        for (int i = 0; i < 11; ++i) h = (h ^ k[i]) * 1099511628211ull;
        hash[l] = h;
        dupOf[l] = -1;
        for (int j = 0; j < l && dupOf[l] < 0; ++j)
            if (((uniq >> j) & 1u) && hash[j] == h && std::memcmp(key[j], k, sizeof key[j]) == 0) dupOf[l] = j;
        step::CollisionCache *cc = caches != nullptr ? caches[l] : nullptr;
        if (dupOf[l] < 0) {
            uniq |= 1u << l;
            ++stats_.lanes;
            single_->DetectCached(s, cc, out[l]);
            continue;
        }
        ++stats_.dupLanes;
        const step::SubstepContacts &o = out[dupOf[l]];
        out[l].n = o.n;
        out[l].overflow = o.overflow;
        std::memcpy(out[l].c, o.c, sizeof(o.c[0]) * static_cast<std::size_t>(o.n));
        ExactCache *ec = cc != nullptr && cc->kind == &ExactCache::kKind ? static_cast<ExactCache *>(cc) : nullptr;
        if (ec == nullptr) continue;
        if (ec->flushEveryCall) ec->Flush();
        ExactCache::Last &la = ec->last;   // DetectPose's last answer
        la.owner = id;
        std::memcpy(la.key, k, sizeof key[l]);
        la.out.n = o.n;
        la.out.overflow = o.overflow;
        std::memcpy(la.out.c, o.c, sizeof(o.c[0]) * static_cast<std::size_t>(o.n));
    }
}

void ExactCollisionLanes::DetectLanesShared(int L, const model::TwinState *const *poses, std::uint32_t mask, step::CollisionCache *const *caches,
                                      step::SubstepContacts *out) const {
    ++stats_.calls;
    const std::uint64_t id = single_->Id();
    // ---- 1. per lane: cache (flush, last answer), shape poses, boxes, union ----
    ShapePose shapes[kMaxLanes][8];
    std::uint32_t key[kMaxLanes][11];
    ExactCache *ec[kMaxLanes];
    std::uint32_t todo = 0u;
    for (int l = 0; l < L; ++l) {
        if (!((mask >> l) & 1u)) continue;
        const model::TwinState &s = *poses[l];
        step::CollisionCache *cc = caches != nullptr ? caches[l] : nullptr;
        ec[l] = cc != nullptr && cc->kind == &ExactCache::kKind ? static_cast<ExactCache *>(cc) : nullptr;
        if (ec[l] != nullptr && ec[l]->flushEveryCall) ec[l]->Flush();
        std::memcpy(key[l], s.pos, 12);
        std::memcpy(key[l] + 3, s.quat, 16);
        std::memcpy(key[l] + 7, s.absorb, 16);
        if (ec[l] != nullptr) {   // DetectPose: the same pose bits as the lane's last call, the same answer
            const ExactCache::Last &la = ec[l]->last;
            if (la.owner == id && std::memcmp(key[l], la.key, sizeof key[l]) == 0) {
                out[l].n = la.out.n;
                out[l].overflow = la.out.overflow;
                std::memcpy(out[l].c, la.out.c, sizeof(out[l].c[0]) * static_cast<std::size_t>(la.out.n));
                ++ec[l]->lastHits;
                ++stats_.lastHits;
                continue;
            }
        }
        todo |= 1u << l;
    }
    if (todo == 0u) return;
    // lanes at the same pose bits get the same contacts: search candidates from one snapshot share their
    // trajectory up to their first change, so detect each distinct pose once
    int dupOf[kMaxLanes];
    std::uint32_t uniq = 0u;
    for (int l = 0; l < L; ++l) {
        if (!((todo >> l) & 1u)) continue;
        dupOf[l] = -1;
        for (int j = 0; j < l && dupOf[l] < 0; ++j)
            if (((uniq >> j) & 1u) && std::memcmp(key[j], key[l], sizeof key[l]) == 0) dupOf[l] = j;
        if (dupOf[l] < 0) {
            uniq |= 1u << l;
            if (sharedMemo_) {
                const model::TwinState &s = *poses[l];
                ShapePoses(single_->Car(), s.pos, s.quat, s.absorb, shapes[l]);
            }
        } else {
            ++stats_.dupLanes;
        }
    }
    if (sharedMemo_) {   // the group path: shared memos, cross-lane packets
        const ShapePose *sp[kMaxLanes];
        for (int l = 0; l < L; ++l) sp[l] = shapes[l];
        DetectShapesLanes(L, sp, uniq, out);
    } else {             // the single-run stage per distinct pose, through the lane's own cache
        for (int l = 0; l < L; ++l)
            if ((uniq >> l) & 1u) {
                ++stats_.lanes;
                single_->DetectCached(*poses[l], caches != nullptr ? caches[l] : nullptr, out[l]);
            }
    }
    for (int l = 0; l < L; ++l) {
        if (!((todo >> l) & 1u)) continue;
        if (dupOf[l] >= 0) {
            const step::SubstepContacts &o = out[dupOf[l]];
            out[l].n = o.n;
            out[l].overflow = o.overflow;
            std::memcpy(out[l].c, o.c, sizeof(o.c[0]) * static_cast<std::size_t>(o.n));
        }
        if (ec[l] == nullptr || (!sharedMemo_ && dupOf[l] < 0)) continue;   // (the single-run stage wrote its own)
        ExactCache::Last &la = ec[l]->last;   // DetectPose's last answer
        la.owner = id;
        std::memcpy(la.key, key[l], sizeof key[l]);
        la.out.n = out[l].n;
        la.out.overflow = out[l].overflow;
        std::memcpy(la.out.c, out[l].c, sizeof(out[l].c[0]) * static_cast<std::size_t>(out[l].n));
    }
}

void ExactCollisionLanes::DetectShapesLanes(int L, const ShapePose *const *shapes, std::uint32_t todo, step::SubstepContacts *out) const {
    const world::WorldPackView &v = *view_;
    ++tick_;
    std::uint64_t t0 = profile_ ? __rdtsc() : 0;
    auto lap = [&](std::uint64_t &acc) { if (profile_) { const std::uint64_t t1 = __rdtsc(); acc += t1 - t0; t0 = t1; } };
    Box3 boxes[kMaxLanes][8], uni[kMaxLanes];
    for (int l = 0; l < L; ++l) {
        if (!((todo >> l) & 1u)) continue;
        ++stats_.lanes;
        for (int k = 0; k < 8; ++k) {   // the prefilter's radius constants follow the poses' radii (fixed for a car)
            const float *r = shapes[l][k].radii;
            if (!(r[0] == scRadii_[k][0] && r[1] == scRadii_[k][1] && r[2] == scRadii_[k][2])) {
                for (int a = 0; a < 3; ++a) scRadii_[k][a] = r[a];
                sc_[k] = cq::MakeShapeConst(r);
            }
        }
        ShapeBox(shapes[l][0], &boxes[l][0]);
        uni[l] = boxes[l][0];
        for (int k = 1; k < 8; ++k) {
            ShapeBox(shapes[l][k], &boxes[l][k]);
            Grow(&uni[l], boxes[l][k]);
        }
    }
    lap(stats_.tBoxes);
    // ---- 2-3. per lane, in lane order: a memo whose box contains the lane's union (else a refill
    // for the group of the remaining lanes it starts), the box test + prefilter, pairs with FV frames ----
    frames_.clear();
    pairs_.clear();
    std::uint32_t pending = todo;
    nRot_ = 0;
    for (int l = 0; l < L; ++l) {
        if (!((todo >> l) & 1u)) continue;
        pending &= ~(1u << l);
        PoolMemo *pm = nullptr;
        if (own_[l].have && Contains(own_[l].q.box, uni[l])) pm = &own_[l];
        for (int m = 0; m < kPool && pm == nullptr; ++m)
            if (pool_[m].have && Contains(pool_[m].q.box, uni[l])) pm = &pool_[m];
        if (pm == nullptr) {   // refill: a shared memo for this lane and the pending lanes that fit, else the lane's own
            Box3 g = uni[l];
            int members = 1;
            for (int j = l + 1; j < L; ++j) {
                if (!((pending >> j) & 1u)) continue;
                const Box3 t = Union(g, uni[j]);
                if (t.hi[0] - t.lo[0] <= kGroupExtent && t.hi[1] - t.lo[1] <= kGroupExtent && t.hi[2] - t.lo[2] <= kGroupExtent) {
                    g = t;
                    ++members;
                }
            }
            if (members >= 2) {
                int mi = 0;
                for (int m = 1; m < kPool; ++m)
                    if (!pool_[m].have || (pool_[mi].have && pool_[m].used < pool_[mi].used)) mi = m;
                pm = &pool_[mi];
            } else {
                pm = &own_[l];
            }
            float center[3];
            for (int a = 0; a < 3; ++a) center[a] = 0.5f * (g.lo[a] + g.hi[a]);
            Box3 big = g;
            if (pm->have)   // extend along the motion since the memo's last refill (only affects reuse)
                for (int a = 0; a < 3; ++a) {
                    const float d = std::max(-4.0f, std::min(4.0f, (center[a] - pm->center[a]) * kLead));
                    if (d > 0.0f) big.hi[a] += d; else big.lo[a] += d;
                }
            const Box3 box = Inflate(big, kMargin);
            float origin[3];
            for (int a = 0; a < 3; ++a) {
                origin[a] = 0.5f * (box.lo[a] + box.hi[a]);
                pm->center[a] = center[a];
            }
            cq::BuildMemo(v, single_->Bvh(), single_->Fast(), single_->ModelAbs(), box, origin, &pm->q);
            pm->have = true;
            ++stats_.memoRefills;
        }
        pm->used = tick_;
        cq::Memo &q = pm->q;
        stats_.memoEntries += q.n;
        lap(stats_.tMemo);
        cq::ShapeFilter sf[8];
        for (int k = 0; k < 8; ++k) cq::MakeShapeFilter(q, shapes[l][k].world.r, shapes[l][k].world.t, sc_[k], boxes[l][k], &sf[k]);
        lap(stats_.tShape);
        cq::Filter(q, sf, &lists_[l], true, &uni[l]);
        lap(stats_.tFilter);
        for (int k = 0; k < 8; ++k) {
            const ShapePose &sp = shapes[l][k];
            std::uint32_t curInst = 0xFFFFFFFFu;
            const std::uint32_t ns = lists_[l].n[k];
            stats_.survivors += ns;
            for (std::uint32_t j = 0; j < ns; ++j) {
                const std::uint32_t i = lists_[l].idx[k][j];
                const cq::Ref &rf = q.ref[i];
                if (rf.instance != curInst) {   // FV's transforms for this (lane, shape, instance); rotation parts shared
                    curInst = rf.instance;
                    exact::FvRot *hit = nullptr;   // key: the shape's rotation and radii bits (compared in place)
                    for (int r = 0; r < nRot_ && hit == nullptr; ++r)
                        if (rot_[r].instance == rf.instance && std::memcmp(rot_[r].key, sp.world.r, 36) == 0 &&
                            std::memcmp(rot_[r].key + 9, sp.radii, 12) == 0)
                            hit = &rot_[r];
                    if (hit == nullptr) {
                        hit = &rot_[nRot_ < kRot ? nRot_++ : (nextRot_++ % kRot)];
                        exact::MakeFvRot(sp.world.r, sp.radii, v.instances[rf.instance].iso, rf.instance, hit);
                    } else {
                        ++stats_.sharedRot;
                    }
                    frames_.emplace_back();
                    exact::FvFrameFrom(*hit, sp.world.t, &frames_.back());
                    ++stats_.frames;
                }
                pairs_.push_back(exact::Pair{static_cast<std::uint32_t>(frames_.size() - 1), rf.modelTri, q.tri[i],
                                             static_cast<std::uint32_t>(8 * l + k)});
            }
        }
        lap(stats_.tFrames);
    }
    // ---- 4. decisions and payloads, packets across lanes and shapes ----
    raw_.clear();
    exact::CollidePackets(frames_.data(), nullptr, verts_.data(), pairs_.data(), pairs_.size(), &raw_);
    stats_.raw += raw_.size();
    lap(stats_.tPackets);
    // ---- 5. per (lane, shape): merge and fill, in FV order ----
    std::size_t r0 = 0;
    for (int l = 0; l < L; ++l) {
        if (!((todo >> l) & 1u)) continue;
        step::SubstepContacts &o = out[l];
        o.n = 0;
        o.overflow = false;
        for (int k = 0; k < 8; ++k) {
            const std::uint8_t tag = static_cast<std::uint8_t>(8 * l + k);
            std::size_t r1 = r0;
            while (r1 < raw_.size() && raw_[r1].shape == tag) ++r1;
            merged_.clear();
            for (std::size_t r = r0; r < r1; ++r) if (raw_[r].primary) merged_.push_back(static_cast<int>(r));
            const std::size_t nPrim = merged_.size();
            for (std::size_t r = r0; r < r1; ++r) {
                const exact::Raw &c = raw_[r];
                if (c.primary) continue;
                std::size_t j = 0;
                for (; j < nPrim; ++j) {
                    const exact::Raw &p = raw_[static_cast<std::size_t>(merged_[j])];
                    if ((NearlyEqual(c.K.x, p.K.x) && NearlyEqual(c.K.y, p.K.y) && NearlyEqual(c.K.z, p.K.z)) || kNormalAlignment < exact::Dot(c.N, p.N))
                        break;
                }
                if (j == nPrim) merged_.push_back(static_cast<int>(r));
            }
            for (int idx : merged_) {
                if (o.n >= model::kRespMaxContacts) { o.overflow = true; break; }
                const exact::Raw &c = raw_[static_cast<std::size_t>(idx)];
                const Tri &tr = v.tris[c.tri];
                const world::TriInfo &ti = v.triInfo[c.tri];
                const world::Actor &a = v.ActorOf(tr);
                model::RespContact &rc = o.c[o.n++];
                rc.P[0] = c.P.x; rc.P[1] = c.P.y; rc.P[2] = c.P.z;
                rc.N[0] = c.N.x; rc.N[1] = c.N.y; rc.N[2] = c.N.z;
                rc.Sep[0] = c.Sep.x; rc.Sep[1] = c.Sep.y; rc.Sep[2] = c.Sep.z;
                rc.peerZ[0] = a.peerZ[0]; rc.peerZ[1] = a.peerZ[1]; rc.peerZ[2] = a.peerZ[2];
                const bool trig = ti.trigger != world::kNoTrigger;
                rc.peerId = trig ? static_cast<std::uint32_t>(ti.trigger) : a.peerId;
                rc.shape = static_cast<std::uint8_t>(k);
                rc.material = static_cast<std::uint8_t>(tr.meta & kMetaMaterialMask);
                rc.flags = static_cast<std::uint8_t>((c.primary ? unsigned(model::kRespPrimary) : 0u) | (trig ? unsigned(model::kRespTrigger) : 0u));
                rc.pad = 0u;
            }
            r0 = r1;
        }
    }
    lap(stats_.tMerge);
}

}  // namespace collide
}  // namespace tw
