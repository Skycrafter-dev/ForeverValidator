// Contact-atlas collision stage (E, gen2 S1; design-gen2-e.md §3.3, gen2-E-synthesis.md). FV-free.
//
// Per substep, per car shape, a decision that needs no history:
//   free  the shape's column lists no plate and no loose triangle within the shape's reach: no contact
//   T0    on a candidate home plate P (the cell's plates in reach, tried in a fixed score order): the shape is
//         clear of foreign geometry (CellRec gaps at its cover points, or plane tests against the cell's other
//         plates) and P's plane is out of reach or behind the centre: no contact
//   T1    ... P's plane is in reach and the foot point lies strictly inside exactly one member: one FACE in
//         closed form (D's tw_atlas_face.h, the same bits as in every other tier), with a proof that FV's
//         per-shape merge leaves exactly it (no other member reaches the contact disk, or FV's 30-degree rule
//         drops every coplanar feature)
//   fallback  everything else: D's per-plane fallback for that shape (tw_atlas_fb.h: closed-form FACEs, the 30-degree
//         proof, FV's form for features and merge keys), on the shape's BVH candidates
// Output: shape-major (shape 0..7), each shape's merged contacts in FV's per-shape order (as collide::MergeShapeTo).
//
// Reads exactly TwinState::pos, quat and absorb (like collide::ExactCollision). Detect is const and
// thread-safe; AtlasCache (step::CollisionCache) holds only hints, so the output is bit-identical with any
// cache content (cold, warm, flushed every call).
#ifndef TW_ATLAS_STAGE_H
#define TW_ATLAS_STAGE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "atlas_build.h"
#include "atlas_colcache.h"
#include "stages.h"
#include "tw_atlas_face.h"
#include "tw_atlas_fb.h"
#include "tw_collide.h"

namespace tw::atlas {

struct AtlasOptions {
    int maxTries = 4;            // candidate home plates tried per shape
    std::size_t columnBudget = std::size_t(64) << 20;   // build.lazyColumns: C's ColumnCache resident budget (bytes)
    bool exactFallback = false;  // test: B's whole-substep exact stage for the fallback shapes instead of D's S2
    int profile = 0;             // measurement only (wrong output): stop a decision after stage 1 the free test, 2 the candidate
                                 // scan, 3 the plate order, 4 the foreign test, 5 the member search, 6 the closed FACE
    bool planeTests = true;      // T1.5: plane tests against the cell's other plates when a gap test fails
    BuildOptions build = LazyColumns();   // C's lazy column cache by default (identical results; 0.2 s init instead of 3-4 s)
    static BuildOptions LazyColumns() { BuildOptions b; b.lazyColumns = true; return b; }
};

enum Decision : std::uint8_t { kFree = 0, kT0 = 1, kT1 = 2, kFallback = 3 };

// why a shape went to the fallback (stats)
enum FallbackWhy : std::uint8_t {
    kFbLoose = 0, kFbNoTile, kFbForeign, kFbReachTie, kFbMember, kFbFaceTie, kFbProof, kFbTries, kFbCount
};

struct AtlasStats {
    std::uint64_t calls = 0, fallbackCalls = 0;
    std::uint64_t shapes[4] = {};          // per Decision
    std::uint64_t why[kFbCount] = {};
    std::uint64_t tries = 0, planeTestsRun = 0, colScanned = 0, fieldTests = 0;
};

class AtlasCache final : public step::CollisionCache {
public:
    AtlasCache();
    void Flush() override;
    collide::ExactCache exact;             // the fallback's memo
    struct TileHint { std::uint32_t plate = kNoPlate; std::int32_t tx = 0, ty = 0; const Tile *t = nullptr; };
    TileHint hint[8][4];                   // per shape, per lookup slot (3 cover points, the foot point)
    std::uint64_t owner = 0;
    static const char kKind;
};

class AtlasCollision final : public step::CollisionStage {
public:
    // The view (and the pack behind it) must outlive the stage.
    bool Init(const world::WorldPackView &view, const collide::CarShapes &car, std::string *error, const AtlasOptions &opt = {});

    void Detect(const model::TwinState &s, step::SubstepContacts &out) const override;
    std::unique_ptr<step::CollisionCache> NewCache() const override;
    void DetectCached(const model::TwinState &s, step::CollisionCache *cache, step::SubstepContacts &out) const override;
    void DetectPose(const float pos[3], const float quat[4], const float absorb[4], step::SubstepContacts &out,
                    step::CollisionCache *cache = nullptr) const;
    // core: the 8 ellipsoids at given world poses (also used against recorded FV poses)
    void DetectShapes(const collide::ShapePose shapes[8], step::SubstepContacts &out, AtlasCache *cache = nullptr,
                      AtlasStats *stats = nullptr, Decision *decisions = nullptr) const;
    // Whole-tick contact-free proof (the airborne split, car-force agent's M3): true when no triangle can touch any
    // car shape at any pose of this tick, given the tick-start state and a bound on the tick's travel:
    //   |v| dt + aMax dt^2 / 2 + min(100, 1.05 |w| + 0.1) dt (R_car + 0.6 m: the lever about the COM) + 0.25 m (wheel
    //   absorb travel), per axis around the 8 shapes' boxes;
    // the swept box is tested against the wheel-class column unions (every column the box meets). Exact if the bound
    // holds: the caller re-checks the real displacement after each substep's integration, before its detection.
    bool TickClear(const model::TwinState &tickStart, float dt, float aMax = 40.0f) const;
    bool TickClearShapes(const collide::ShapePose shapes[8], const float pos[3], const float vel[3], const float angVel[3], float dt,
                         float aMax = 40.0f) const;
    // one shape's decision; T1 fills *c (all RespContact fields)
    Decision Decide(const collide::ShapePose &sp, int k, model::RespContact *c, AtlasCache &cache, AtlasStats *stats,
                    FallbackWhy *why) const;

    const Atlas &GetAtlas() const { return atlas_; }
    const collide::ExactCollision &Exact() const { return exact_; }
    std::uint64_t Id() const { return id_; }

private:
    collide::ExactCollision exact_;        // the world view, BVH4 and FV-form data for the fallback and the atlas build
    Atlas atlas_;
    Fallback fb_;                          // D's S2 (its FaceTri table, per world triangle, is the one T1 uses too)
    AtlasOptions opt_{};
    std::uint64_t id_ = 0;
    const world::WorldPackView *view_ = nullptr;
    // the column data: the whole grid, or C's lazy ColumnCache (build.lazyColumns)
    std::unique_ptr<ColumnCache> colCache_;
    float colX0_ = 0.0f, colZ0_ = 0.0f;
    std::uint32_t colNx_ = 0, colNz_ = 0;
    struct ColView {
        const ColumnEntry *e = nullptr;
        std::uint32_t n = 0;
        const std::int32_t *u = nullptr;
        std::uint32_t nu = 0;
        ColumnCache::SlotRef ref;      // cache form
        std::uint32_t globalFirst = 0; // whole-grid form: the slot's first entry
        float cx0 = 0.0f, cz0 = 0.0f;  // the dilated column's min corner
    };
    bool Column(const float x[3], int cls, ColView *v) const;   // false outside the grid
    const std::uint32_t *ColumnMembers(const ColView &v, std::uint32_t i, std::uint32_t *count, bool *overlap) const;
    int Reader() const;                                          // this thread's ColumnCache reader
    enum TryResult { kTryT0, kTryT1, kTryFail };
    TryResult TryPlate(const collide::ShapePose &sp, int k, int ci, const std::uint32_t *cand, const float *candZ, const float *candH, int ncand,
                       model::RespContact *c, AtlasCache &cache, AtlasStats *stats, FallbackWhy *why) const;
    const Tile *Lookup(AtlasCache &cache, int k, int slot, std::uint32_t plate, std::int32_t tx, std::int32_t ty) const;
};

}  // namespace tw::atlas

#endif
