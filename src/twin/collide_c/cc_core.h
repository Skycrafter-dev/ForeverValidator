// Architect C collision prototype: feature caching with validity regions and
// clearance-bounded skipping (docs/twin/collision-c.md).
//
// World (cold, built once per map): world-space convex coplanar patches. A
// patch is a set of edge-adjacent source triangles with the same material
// bits whose vertices lie within a thin slab; it keeps its member triangles
// (FV stored order, own plane, own in-plane edge normals, neighbour links) and
// its convex outline. A small binary BVH over patch boxes answers re-queries.
//
// Car cache (hot, per candidate): per car shape,
//   - tracked patches: evaluated every pose in closed form. Plane slab not
//     reached -> no contact from any member; reached -> the contact point P
//     is located by walking member triangles; P inside member T with the
//     contact disk clear of T's edges (or FV's 30-degree merge provably
//     dropping every coplanar feature) -> FV's FACE in closed form; anything
//     else -> FV's exact test on the members (B's tw_exact.h);
//   - sleepers: every other patch in the query box, skipped until the shape's
//     odometer (sum of |dt| + ||dM||_F over evaluated poses) uses up a
//     conservative clearance (AABB gap first, then a separating-axis margin);
//     a sleeper whose margin drops below `promote` becomes tracked;
//   - one car-level BVH query whenever a shape's odometer has used the query
//     horizon L since the last query.
// FV's per-shape merge (board 20 §3.4) runs only when a shape has both a FACE
// and a feature contact.
#ifndef CC_CORE_H
#define CC_CORE_H

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tw_exact.h"
#include "tw_world.h"

#ifndef CC_STATS
#define CC_STATS 1
#endif
#ifndef CC_SPLIT
#define CC_SPLIT 0
#endif
#ifndef CC_MICROB
#define CC_MICROB 0
#endif
#ifndef CC_FINE
#define CC_FINE 0
#endif

namespace cc {

using tw::Box3;
constexpr int kShapes = tw::kShapes;
constexpr int kProxy = 8;   // slot of the body proxy: a car-frame box around body shapes 0-3
constexpr int kSlots = 9;   // cache slots: wheels 4-7 and the body proxy (body shapes 0-3 hold no lists)
constexpr std::uint32_t kMetaKeyMask = tw::kMetaMaterialMask | tw::kMetaTrigger | tw::kMetaCarIgnores;

struct PEdge { float u[3]; float a[3]; };  // convex outline edge: in-plane outward unit normal, a point on it

struct Member {          // one source triangle of a patch (FV stored order)
    float v0[3], n[3];   // vertex 0 (subtract-first origin), own unit normal
    float u[3][3];       // in-plane outward unit normal of edge i (v_i -> v_i+1)
    float a[3][3];       // edge origins v_i
    std::int32_t nb[3];  // member (global index) across edge i inside the patch, -1 = patch boundary
    std::uint32_t tri, meta, patch;
};

enum : std::uint32_t { kPatchExactOnly = 1u };

struct Patch {
    float n[3], ref[3];  // plane through ref with unit normal n
    float tau;           // every member vertex lies within tau of the plane (incl. float slack)
    float alpha;         // max |n_member - n|
    float cen[3], rad;   // bounding sphere of the member vertices
    Box3 box;
    std::uint32_t meta, flags;
    std::uint32_t firstEdge, edgeCount, firstMember, memberCount;
};

}  // namespace cc
#include "tw_accel.h"
namespace cc {

struct PatchWorld {
    std::vector<Patch> patches;
    std::vector<PEdge> edges;
    std::vector<Member> members;
    std::vector<std::uint32_t> triPatch;
    std::size_t exactOnly = 0, multi = 0, maxMembers = 0, enclosure = 0;
    bool precise = false;
    double buildMs = 0;
    void Build(const tw::World &w, const tw::SourceScene *src, double angTol, double distTol, std::uint32_t maxMembersPerPatch);
    tw::Bvh4 bvh;  // B's 4-wide BVH over the patch boxes (one pseudo-triangle per patch)
    void BuildBvh();
    std::size_t Bytes() const;
};

// One contact as the twin hands it on (FV merged-buffer fields + provenance).
struct Emitted {
    tw::Contact c;
    std::uint32_t tri;
    std::uint8_t material, primary;
};

using KSet = std::array<std::uint64_t, 4>;  // materials of one shape's merged contacts
inline void KAdd(KSet &k, int m) { k[m >> 6] |= 1ull << (m & 63); }

struct Stats {
    std::uint64_t poses = 0, queries = 0, queryPatches = 0, queryNodes = 0;
    std::uint64_t expireScans = 0, marginEvals = 0, promotions = 0, demotions = 0;
    std::uint64_t trackedEvals = 0, emptyBehind = 0, emptyFar = 0, emptyOutside = 0, faceClosed = 0, faceAngle = 0;
    std::uint64_t walkSteps = 0;
    std::uint64_t fbBorder = 0, fbOutside = 0, fbWalk = 0, fbMember = 0, fbDeep = 0, fbExactOnly = 0;  // exact fallbacks by reason
    std::uint64_t exactTests = 0, exactHits = 0, merges = 0;
    std::uint64_t sleepersHeld = 0, trackedHeld = 0, maxTracked = 0;
    std::uint64_t perShapeTracked[kShapes] = {}, perShapeFallback[kShapes] = {}, perShapeMargins[kShapes] = {};
    std::uint64_t probes = 0, probeHorizon = 0, probeMargins = 0, resets = 0;
    std::uint64_t faceFast = 0, axisMismatch = 0, slabCertified = 0, sweeps = 0, proxyBodySlab = 0, steadyPoses = 0;
    std::uint64_t busyBy[6] = {};  // poses with: query, car-level scan, slot scan, wheel full evaluation, proxy evaluation, body full evaluation
    std::uint64_t boxRecerts = 0, carHeld = 0, carScans = 0, carWakes = 0, carRecerts = 0, distributed = 0;
    std::uint64_t proxyEvals = 0, proxyCertified = 0, proxyExpand = 0, proxyViolations = 0;
};

// Pose of one shape (world).
struct Pose {
    float t[3];
    float M[3][3];  // x = M u + t, M = axes * diag(r)
    float rmax, rmin;
    mutable Box3 box_;           // AABB inflated by 1e-3 (B's kEps), computed on first use (B())
    mutable bool boxValid = false;
    const Box3 &B() const {
        if (!boxValid) {
            for (int i = 0; i < 3; ++i) {
                const float h = (std::fabs(M[i][0]) + std::fabs(M[i][1])) + std::fabs(M[i][2]);
                box_.lo[i] = t[i] - h - 1e-3f;
                box_.hi[i] = t[i] + h + 1e-3f;
            }
            boxValid = true;
        }
        return box_;
    }
    tw::ShapeMap sm;
    bool smValid;
    bool isBox;     // body proxy: M = R_car diag(half extents), support = |M^T u|_1
    float cM[3][3]; // body proxy: box around the 4 body centres, same centre (the back-face certificate needs every centre behind)
    const tw::Shape *src;
};

// A clearance certificate: no member of the patch can be touched while the shape's motion along the
// certificate's direction stays below m. Motion is bounded with per-axis odometers (O_a = sum |dt_a|,
// O_M = sum ||dM||_F over evaluated poses): |w.dt| + ||dM||_2 <= sum_a |w_a| dO_a + dO_M.
enum CertKind : int { kCertBox = 0, kCertPlane = 1, kCertEdge = 2 };
struct Cert { float w[3]; float m; float score; int kind; };

// Sleeping patches of one shape, SoA, padded to 16 lanes (padding never wakes).
struct SleepList {
    std::vector<float> wx, wy, wz, base;
    std::vector<float> lo[3], hi[3];  // the patch's AABB (vector re-certification by AABB gap on wake)
    std::vector<std::uint32_t> pid;
    std::size_t n = 0;
    void Clear();
    void Push(std::uint32_t p, const float w[3], float b, const Box3 &box);
    void Remove(std::size_t i);
};

class CarCache {
public:
    struct ShapeState;
    const PatchWorld *pw = nullptr;
    const tw::World *world = nullptr;
    float L = 2.0f;           // query horizon (m of the largest axis odometer + rotation odometer)
    float promote = 3.0f;     // track a patch whose best certificate is predicted to last fewer poses than this
    float reserve = 0.0f;     // odometer kept in hand for the predictor's probe poses (set per tick by the caller)
    float Lposes = 8.0f;      // query box: inflated per axis by L + Lposes x recent motion per pose along that axis
    void SetEarlyOut(float poses) { earlyOut_ = poses; }
    Stats st;   // collision passes
    Stats pst;  // predictor probes

    struct Track { std::uint32_t patch; std::int32_t member; std::uint32_t stamp; };  // stamp: pose resolved by the FACE kernel
    struct ShapeState {
        Pose pose;
        float O[4] = {0, 0, 0, 0};     // odometers (re-based at sweeps once past 16 m): |dt_x|, |dt_y|, |dt_z| sums, ||dM||_F sum
        float Oq[4] = {0, 0, 0, 0};    // odometers at the last query (horizon check)
        float rate[4] = {0, 0, 0, 0};  // recent motion per pose (direction choice only)
        SleepList sleep;
        float budget = 0;              // no sleeper can wake before the odometers grow by this much (|dO_xyz|_2 + dO_M)
        std::vector<Track> tracked;
        // SoA copy of the tracked patches' slabs for the 16-lane slab pass (same order as tracked, padded):
        // ref xyz, n xyz, tau (1e30 for exact-only patches), alpha, |ref - cen|_1 + rad
        std::vector<float> sl[9];
        std::vector<Emitted> raw, merged;
    };
    ShapeState s[kSlots];
    float proxyC[3] = {0, 0, 0}, proxyH[3] = {0, 0, 0};  // body proxy box in the car frame
    float proxyHC[3] = {0, 0, 0};                         // half extents of the box of the body centres around proxyC

    void Reset();
    void Step(const tw::Substep &sub);  // collision at this pose; updates the cache
    // Contact-set predictor: does K (materials per shape) at an off-chain pose differ from K0? The cache is
    // not modified. out == nullptr: stop at the first differing shape; else fill all 8 sets.
    bool ProbeDiffers(const tw::Substep &pose, const KSet K0[kShapes], KSet *out);
    void KOf(KSet out[kShapes]) const;
    void Dump(int shape);  // diagnostic: the shape's tracked and soon-waking patches with their certificates
    void MicroBench(double ghz);  // CC_SPLIT: intrinsic cost of Certify / EvalPatch / Expire on the current cache

private:
    bool init_ = false;
    std::uint32_t poseCount_ = 0;
    float qL_[3] = {0, 0, 0};  // per-axis inflation of the last query
    float earlyOut_ = 1e30f;   // Certify returns the AABB-gap certificate alone when it is predicted to last this many poses
    // car level: patches in the query box that no slot needs yet, certified by their AABB gap to the union of
    // the slot boxes along one world axis; per-axis odometer = sum over poses of the largest slot increment
    SleepList car_;
    // patch -> held somewhere (car level or all 5 slots); rebuilt at sweeps
    struct Holders {
        std::vector<std::uint32_t> key;
        std::size_t used = 0;
        void Clear() { if (key.empty()) key.assign(4096, 0u); else std::fill(key.begin(), key.end(), 0u); used = 0; }
        void Reserve(std::size_t n) {  // empty table with load <= 1/4 for n keys
            std::size_t cap = 4096;
            while (cap < 4 * n) cap *= 2;
            key.assign(cap, 0u);
            used = 0;
        }
        float Load() const { return key.empty() ? 1.0f : float(used) / float(key.size()); }
        bool Insert(std::uint32_t pid) {  // true if new
            if (key.empty()) Clear();
            const std::size_t mask = key.size() - 1;
            std::size_t h = (pid * 2654435761u) & mask;
            for (;;) {
                if (key[h] == pid + 1) return false;
                if (key[h] == 0) { key[h] = pid + 1; ++used; return true; }
                h = (h + 1) & mask;
            }
        }
    } holders_;
    std::uint32_t queriesSinceSweep_ = 0;
    Box3 prevQ_ = tw::EmptyBox();
    bool havePrevQ_ = false;
    std::size_t heldAtSweep_ = 0;
    void Sweep(const Box3 &q);
    float lastCarPos_[3] = {0, 0, 0};
    const tw::Substep *lazySub_ = nullptr;
    bool bodiesValid_ = false;
    std::uint64_t fineLast_ = 0;
public:
    unsigned dbgSkip_ = 0;
    bool busy_ = false;
    unsigned busyBits_ = 0;  // this pose left the steady path (query, expiry scan, full evaluation, proxy work)  // micro-benchmark only: skip parts of Step (1 wheel poses, 2 proxy, 4 merge, 8 FACE kernel, 16 other tracked)
private:
    float prevRot_[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    struct FaceItem { const Pose *pose; std::uint32_t patch; std::int32_t member; std::uint8_t slot; std::uint32_t index; };
    std::vector<FaceItem> batch_;
    std::vector<Emitted> batchRes_;
    std::vector<std::uint8_t> batchOk_;
    int FaceBatch(const FaceItem *items, int n, Emitted *res, bool *ok);
    // steady state: per wheel lane, the patch and member of its last FACE, cached SoA for the 4-lane kernel
    alignas(16) float primM_[26][4] = {};
    std::uint32_t primPatch_[4] = {0, 0, 0, 0};
    std::int32_t primMember_[4] = {-1, -1, -1, -1};
    bool primValid_[4] = {false, false, false, false};
    bool sharedWheelM_ = false;  // the 4 wheels' M are identical (axes = car columns, equal radii)
    void SetPrimary(int lane, std::uint32_t patch, std::int32_t member);
    void TrackPush(ShapeState &S, std::uint32_t pid);
    void TrackRemove(ShapeState &S, std::size_t i);
    // 16-lane slab pass over a slot's tracked list: cert[i] = 1 when the slab certifies "no member touched"
    // (centre behind every member plane, or slab out of reach), m[i] / sc[i] its margin and predicted life
    void SlabPass(ShapeState &S, const Pose &p, std::vector<std::uint8_t> &cert, std::vector<float> &mg, std::vector<float> &sc);
    bool BodiesCertified(Pose *const *bodies, std::uint32_t pid) const;
    std::vector<std::uint8_t> slCert_;
    std::vector<float> slM_, slS_;
    unsigned PrimaryKernel(const Pose *const *poses, float (*out)[4]);
    static void WheelPose(const tw::Shape &sh, Pose *p);
    void CarBox();
    float carO_[4] = {0, 0, 0, 0}, carRate_[3] = {0, 0, 0}, carBudget_ = 0;
    Box3 carBox_ = tw::EmptyBox();
    void ExpireCar();
    void Distribute(std::uint32_t pid);  // hand a car-level patch to every slot (slot-level AABB-gap certificates)
    std::vector<std::uint32_t> removed_;
    Stats *cur_ = &st;
    std::vector<std::uint32_t> qlist_;

    std::vector<Emitted> praw_, pmerged_, praw4_[4];
    Pose probe_[kSlots], qb_[4];
    std::vector<std::uint32_t> extra_[kSlots];
    void Query();
    void Expire(int k);
    // Best certificate (largest predicted lifetime) for the patch at this pose; m <= 0: may touch.
    Cert Certify(const Pose &p, const Patch &pa, const float rate[4]);
    enum Eval { kEmptyBehind, kEmptyFar, kEmptyOutside, kContact };
    Eval EvalPatch(Pose &p, std::uint32_t patch, std::int32_t &hint, std::vector<Emitted> &out, int shape, const float rate[4], Cert *cert);
    void ExactMembers(Pose &p, const Patch &pa, std::vector<Emitted> &out);
    // body proxy: certified (no body shape can touch the patch) or the 4 body shapes evaluated into raw[0..3]
    Eval EvalProxyInto(const Pose &box, std::uint32_t pid, Pose *const *bodies, std::vector<Emitted> *const *raw, const float rate[4], Cert *cert,
                       const tw::Substep *lazy, bool *valid);
    void ProxyPose(const tw::Substep &sub, Pose *p) const;
};

void SetupPose(const tw::Shape &sh, Pose *p);
void MakeCert(const float n[3], float mg, const float rate[4], Cert *c);
void MergeShape(const std::vector<Emitted> &in, std::vector<Emitted> *out);
bool FeaturesDropped(const tw::ShapeMap &m, const float n[3], const float Mn[3], float h, float d);

// Hot-bytes accounting (stats builds): distinct 64-byte lines touched per tick.
#if CC_STATS
void Touch(const void *p, std::size_t bytes, int cls);
#define CC_TOUCH(p, b, c) ::cc::Touch((p), (b), (c))
#define CC_CNT(x) (++(x))
#define CC_ADD(x, v) ((x) += (v))
#else
#define CC_TOUCH(p, b, c) ((void)0)
#define CC_CNT(x) ((void)0)
#define CC_ADD(x, v) ((void)0)
#endif
enum TouchClass { kTouchHot = 0, kTouchQuery = 1, kTouchExact = 2 };
struct TouchCounter {
    std::vector<std::uint64_t> slots;
    std::uint64_t gen = 1;
    std::size_t lines[3] = {0, 0, 0};
    void Clear();
};
extern TouchCounter g_touch;
extern std::uint64_t g_split[8];  // CC_SPLIT builds: rdtsc per phase (setup, query, expire, tracked + merge)

}  // namespace cc

#endif
