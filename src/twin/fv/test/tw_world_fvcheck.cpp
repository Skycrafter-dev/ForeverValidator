// tw_world_fvcheck: the M2 world package against FV, on real replays.
//
//   tw_world_fvcheck PACKS [--out DIR [--keep]] [--probes N] [--quiet] REPLAY...
//
// Loads each replay through FV's Reference backend (PhysicsSandbox; the load
// runs FV's prestart), compiles the package (tw::fv::CompileWorldPack) from
// FV's flat scene export, and checks it against what FV itself uses:
//  1. triangles: FV's live static collision records of the car's target
//     groups (CHmsCollisionManager::SZone, the records the Reference
//     detection traverses, in traversal order: group, record preorder, mesh
//     octree leaf), expanded with FV's own GmVec3::SetMult. Count, order
//     (package fvOrder), every vertex bit, material id, group/trigger flag,
//     stored plane, model triangle index, record iso;
//  2. actors: peerZ bit-equal to FV GetLocation()->GetDir (the reference
//     booster axis), peerId == corpus Id().Value(), installation order;
//     each FV corpus maps to one actor;
//  3. race: trigger table == FV's corpus collection (role, current
//     transform, slot, spawn iso; A's race_map_from_actors recipe), trigger
//     id of every group-1 triangle == its corpus's rank among
//     IsRaceTriggerMobil corpora, spawn table (spawn 0 = previousSpawnLocation,
//     GmQuat::Set quaternions), requiredCps / requiredLaps == FV progress;
//  4. water: tw::model::WaterZone over the package grid == FV's zone
//     (SZone::waterZone.AcceptsRegion) on random and cell-centre probes;
//  5. package: deterministic (two compiles, same hash), save/load round
//     trip, full validation; B's bench world (MakeBenchWorld) == B's
//     prototype CompileTwinWorld on the same scene.
// Hooks: GNU ld --wrap on four FV symbols (CMakeLists.txt); FV is unmodified.
#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/native.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/core/gm_types.h"
#include "engine/game/game_ctn_block.h"
#include "engine/game/game_ctn_block_info.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/physics/geometry/gm_surface.h"
#include "engine/physics/geometry/plug_surface.h"
#include "engine/rendering/plug_tree.h"
#include "engine/scene/replay_static_scene_corpuses.h"
#include "engine/scene/scene_mobil.h"
#include "engine/scene/scene_vehicle_water_zone.h"
#include "simulation/backends/cuda/cuda_scene_layout.h"
#include "simulation/backends/cuda/cuda_static_configuration.h"
#include "simulation/runtime/replay_simulation_runtime.h"
#include "simulation/runtime/replay_simulation_session.h"
#include "src/twin/common/twin_fv_state.h"
#include "car_force_consts.h"
#include "compile_world.h"
#include "tw_world_compile.h"
#include "tw_world_pack.h"

namespace fv = forevervalidator;
namespace fvx = forevervalidator::experimental;
namespace w = tw::world;
using tw::race::TwSpawn;
using tw::race::TwTrigger;

// ---- private member access (explicit instantiation ignores access checks) -------------------
namespace rob {
template <typename Tag, typename Tag::type M>
struct Robber {
    friend typename Tag::type Get(Tag) { return M; }
};
using Zone = CHmsCollisionManager::SZone;
using Group = CHmsCollisionManager::SGroup;
struct ZoneGroups {
    using type = std::array<Group, CHmsCollisionManager_GroupCount> Zone::*;
    friend type Get(ZoneGroups);
};
struct GroupTrees {
    using type = GmOctree<CHmsCollisionManager::SColOctreeCell> Group::*;
    friend type Get(GroupTrees);
};
struct GroupAgainst {
    using type = std::vector<Group::SAgainstGroup> Group::*;
    friend type Get(GroupAgainst);
};
template struct Robber<ZoneGroups, &Zone::groups>;
template struct Robber<GroupTrees, &Group::staticTrees>;
template struct Robber<GroupAgainst, &Group::againstGroups>;
}  // namespace rob

namespace {

std::uint32_t Bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
void PutIsoWords(std::uint32_t *w12, const GmIso4 &iso) {
    const GmVec3 *c[4] = {&iso.rotation.basisX, &iso.rotation.basisY, &iso.rotation.basisZ, &iso.translation};
    for (int k = 0; k < 4; ++k) {
        w12[3 * k] = Bits(c[k]->x);
        w12[3 * k + 1] = Bits(c[k]->y);
        w12[3 * k + 2] = Bits(c[k]->z);
    }
}
double Ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---- stats -------------------------------------------------------------------------------
struct Stats {
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> byName;  // pass, fail
    std::uint64_t failures = 0;
    bool quiet = false;
    void Check(bool ok, const char *name, const std::string &detail) {
        auto &e = byName[name];
        if (ok) {
            ++e.first;
            return;
        }
        ++e.second;
        ++failures;
        if (e.second <= 5u || !quiet) std::printf("  FAIL %s: %s\n", name, detail.c_str());
    }
} g_stats;

// ---- hook state ---------------------------------------------------------------------------
struct FvTri {
    std::uint32_t v[9];       // world vertex bits (SetMult of the mesh vertex by the record iso)
    std::uint32_t material;   // SurfaceMaterialIdFromLocalIndex
    std::uint32_t group;      // target group (1..5)
    std::uint32_t meshTri;    // mesh triangle index
    std::uint32_t recordLeaf; // rank of the record among the group's surface records (preorder)
    std::uint32_t n[3], d;    // stored mesh-local plane bits
    std::uint32_t iso[12];    // record location bits
    const CHmsCorpus *corpus;
};
struct FvSnapshot {
    bool taken = false;
    std::vector<FvTri> tris;
    std::uint32_t records = 0, skippedNoCollision = 0, nonMesh = 0, group3Records = 0, againstGroups = 0;
    std::uint32_t movingCorpora = 0, movingPairs = 0;  // non-static corpora of the target groups (car excluded); pairs FV tests
    std::vector<std::string> movingInfo;
    std::string order;  // target group order as FV traverses it
    const rob::Zone *zone = nullptr;
};

struct CollectionTrigger {
    const CHmsCorpus *corpus;
    TwTrigger t;
    bool hasSpawn;
    GmIso4 spawn;
};

std::vector<std::string> g_movingProvenance;  // provenance of non-static models in target groups (--describe)
std::vector<std::string> g_hugeProvenance;    // provenance of models with a collision surface > 2 km (--describe)
bool g_describe = false;
std::unique_ptr<fv::simulation::CudaHostScene> g_scene;
double g_sceneMs = 0.0;
std::string g_sceneDiag;
std::unique_ptr<fv::simulation::CudaHostStaticConfiguration> g_config;
bool g_haveStartSpawn = false;
GmIso4 g_startSpawn{};
FvSnapshot g_fv;
std::vector<CollectionTrigger> g_collTriggers;
std::unordered_map<const CHmsCorpus *, std::uint32_t> g_corpusOrder;  // corpus -> installation order
std::unordered_map<const CHmsCorpus *, std::uint32_t> g_corpusTrigger;  // corpus -> rank among race-trigger corpora
std::uint32_t g_collectionCount = 0u;

void ResetHooks() {
    g_scene.reset();
    g_sceneMs = 0.0;
    g_sceneDiag.clear();
    g_config.reset();
    g_haveStartSpawn = false;
    g_fv = FvSnapshot{};
    g_collTriggers.clear();
    g_corpusOrder.clear();
    g_corpusTrigger.clear();
    g_collectionCount = 0u;
}

void EmitMesh(const CPlugSurface *surface, const GmIso4 &iso, std::uint32_t g, std::uint32_t recordLeaf, const CHmsCorpus *corpus) {
    const GmSurfMesh *mesh = dynamic_cast<const GmSurfMesh *>(surface->Geometry());
    if (mesh == nullptr) {
        ++g_fv.nonMesh;
        return;
    }
    for (std::uint32_t c = 0; c < mesh->OctreeCellCount(); ++c) {
        const auto &cell = mesh->OctreeCell(c);
        if (!cell.ContainsTriangle()) continue;
        const GmSurfMeshTriangle &tri = mesh->Triangle(cell.TriangleIndex());
        FvTri t{};
        for (int k = 0; k < 3; ++k) {
            GmVec3 wv;
            wv.SetMult(mesh->Vertex(tri.vertexIndex[k]), iso);
            t.v[3 * k] = Bits(wv.x);
            t.v[3 * k + 1] = Bits(wv.y);
            t.v[3 * k + 2] = Bits(wv.z);
        }
        t.material = static_cast<std::uint32_t>(surface->SurfaceMaterialIdFromLocalIndex(tri.material));
        t.group = g;
        t.meshTri = cell.TriangleIndex();
        t.recordLeaf = recordLeaf;
        t.n[0] = Bits(tri.normal.x);
        t.n[1] = Bits(tri.normal.y);
        t.n[2] = Bits(tri.normal.z);
        t.d = Bits(tri.planeDistance);
        PutIsoWords(t.iso, iso);
        t.corpus = corpus;
        g_fv.tris.push_back(t);
    }
}

// The surfaces FV's moving-pair recursion (DetectCollisionBetween -> ComputeCollision,
// ComputeCollisionTree1/2RootOnly) can test against a car shape, in the order a shape
// meets them: root and depth-1 nodes need HasWorldBox, deeper nodes AllowsSurfaceCollision.
void EmitMovingTree(const CPlugTree *t, const GmIso4 &parent, int depth, std::uint32_t g, const CHmsCorpus *corpus) {
    if (depth <= 1 ? !t->HasWorldBox() : !t->AllowsSurfaceCollision()) return;
    GmIso4 iso;
    t->ComposeCollisionIso(parent, iso);
    if (const CPlugSurface *surface = t->Surface()) EmitMesh(surface, iso, g, UINT32_MAX, corpus);
    for (u32 i = 0; i < t->GetChildCount(); ++i)
        if (const CPlugTree *c = t->GetChild(i)) EmitMovingTree(c, iso, depth + 1, g, corpus);
}

void SnapshotZone(const rob::Zone *zone, CHmsCorpus *car) {
    g_fv = FvSnapshot{};
    g_fv.taken = true;
    g_fv.zone = zone;
    const auto &groups = zone->*Get(rob::ZoneGroups{});
    const std::uint32_t carGroup = car->Item()->CollisionGroup();
    const rob::Group &grp = groups[carGroup - 1u];
    for (const rob::Group::SAgainstGroup &against : grp.*Get(rob::GroupAgainst{})) {
        const rob::Group *target = against.targetGroup;
        const std::uint32_t g = static_cast<std::uint32_t>(target - groups.data()) + 1u;
        g_fv.order += std::to_string(g);
        ++g_fv.againstGroups;
        for (std::uint32_t i = 0; i < target->NonStaticCorpusCount(); ++i) {
            CHmsCorpus *other = target->NonStaticCorpusAt(i);
            if (other == nullptr || other == car) continue;
            ++g_fv.movingCorpora;
            const bool enabled = against.collisionSchedule.IsEnabled(*car, *other);
            if (enabled) ++g_fv.movingPairs;
            GmBoxAligned box{};
            if (other->CollisionTree() != nullptr) other->CollisionTree()->GetTransformedCollisionBox(*other->LocationIso(), box);
            const auto &pr = other->Item()->GetProperties();
            const auto ord = g_corpusOrder.find(other);
            char buf[400];
            std::snprintf(buf, sizeof buf,
                          "group %u moving corpus %s pair %s: order %d, dyn %u, static %d, loc (%.3f %.3f %.3f), box c (%.1f %.1f %.1f) h (%.1f %.1f %.1f)",
                          g, other == car ? "car" : "other", enabled ? "on" : "off", ord == g_corpusOrder.end() ? -1 : int(ord->second),
                          unsigned(pr.dynamicType), pr.collisionStatic ? 1 : 0, other->LocationIso()->translation.x,
                          other->LocationIso()->translation.y, other->LocationIso()->translation.z, box.center.x, box.center.y,
                          box.center.z, box.halfExtents.x, box.halfExtents.y, box.halfExtents.z);
            g_fv.movingInfo.push_back(buf);
            if (enabled && other->CollisionTree() != nullptr) EmitMovingTree(other->CollisionTree(), *other->LocationIso(), 0, g, other);
        }
        if (target->StaticTreeCount() <= 1u) continue;  // DetectCollisionsCorpus skips it
        const auto &trees = target->*Get(rob::GroupTrees{});
        std::uint32_t leaf = 0u;
        for (std::uint32_t i = 0; i < trees.GetCount(); ++i) {
            const CHmsCollisionManager::SColOctreeCell &rec = trees[i];
            if (!rec.ContainsSurface()) continue;
            const auto &sd = rec.SurfaceData();
            ++g_fv.records;
            if (!sd.tree->AllowsSurfaceCollision()) {  // traversal condition (collision_detection.cpp)
                ++g_fv.skippedNoCollision;
                continue;
            }
            const std::uint32_t recordLeaf = leaf++;
            if (g == 3u) ++g_fv.group3Records;
            EmitMesh(sd.surface, sd.location, g, recordLeaf, sd.corpus);
        }
    }
}

void BuildCollectionTriggers(ReplayStaticCorpusCollection *c) {
    g_collTriggers.clear();
    g_corpusOrder.clear();
    g_corpusTrigger.clear();
    g_collectionCount = c->Count();
    for (std::uint32_t i = 0; i < c->Count(); ++i) {
        const ReplayStaticCorpus *corpus = c->CorpusAt(i);
        if (corpus == nullptr) continue;
        g_corpusOrder[&corpus->Corpus()] = corpus->InstallationOrder();
        if (!corpus->IsRaceTriggerMobil()) continue;
        // A's BuildTriggerTable (src/twin/race/test/tw_race_fvcheck.cpp)
        const CSceneMobil *mobil = corpus->Item().SceneMobilOwner();
        const CGameCtnBlock *block = mobil != nullptr ? mobil->OwningBlock() : nullptr;
        const CGameCtnBlockInfo *info = block != nullptr ? block->BlockInfoRef() : nullptr;
        CollectionTrigger t{};
        t.corpus = &corpus->Corpus();
        t.t.role = info != nullptr ? static_cast<std::uint8_t>(info->RaceRole()) : static_cast<std::uint8_t>(tw::race::kRoleNone);
        t.t.currentTransform = info != nullptr && info->RespawnUsesCurrentTransform() ? 1u : 0u;
        const std::optional<std::uint32_t> slot = block ? c->CheckpointSlotForBlock(block) : std::nullopt;
        t.t.slot = slot.has_value() ? static_cast<std::uint16_t>(*slot) : tw::race::kNoSlot;
        const GmIso4 *spawn = block ? c->SpawnIsoForCheckpointBlock(block) : nullptr;
        t.hasSpawn = spawn != nullptr;
        if (spawn) t.spawn = *spawn;
        g_corpusTrigger[t.corpus] = static_cast<std::uint32_t>(g_collTriggers.size());
        g_collTriggers.push_back(t);
    }
}

}  // namespace

extern "C" {
bool __real__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(ReplaySimulationSession *,
                                                                                          StaticSceneModelCollection *);
bool __wrap__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(ReplaySimulationSession *self,
                                                                                          StaticSceneModelCollection *models) {
    auto scene = std::make_unique<fv::simulation::CudaHostScene>();
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = fv::simulation::BuildCudaHostScene(*models, scene.get(), tw::fv::TwinSceneBuildLimits());
    g_sceneMs = Ms(t0, std::chrono::steady_clock::now());
    if (r == fv::simulation::CudaSceneBuildResult::Success) {
        g_movingProvenance.clear();
        g_hugeProvenance.clear();
        for (const auto &sf : scene->surfaces) {
            const GmVec3 &h = sf.worldBounds.halfExtents;
            if (std::max(h.x, std::max(h.y, h.z)) < 1000.0f || sf.actorIndex >= models->Models().size()) continue;
            const StaticSceneProvenance &pv = models->Models()[sf.actorIndex].Provenance();
            g_hugeProvenance.push_back("actor " + std::to_string(sf.actorIndex) + " purpose " +
                                       std::to_string(scene->actors[sf.actorIndex].purpose) + " descriptor '" + pv.descriptorPath +
                                       "' sceneObject '" + pv.sceneObjectId + "' block '" + pv.blockName + "' type " +
                                       std::to_string(sf.type) + " tris " + std::to_string(sf.triangleCount));
        }
        for (std::size_t i = 0; i < scene->actors.size() && i < models->Models().size(); ++i) {
            const auto &a = scene->actors[i];
            const std::uint32_t g = static_cast<std::uint32_t>(a.itemProperties.collisionGroup);
            if (a.itemProperties.collisionStatic || !(g == 1u || g == 3u || g == 4u)) continue;
            const StaticSceneProvenance &pv = models->Models()[i].Provenance();
            g_movingProvenance.push_back("actor " + std::to_string(i) + " purpose " + std::to_string(a.purpose) + " block '" +
                                         pv.blockName + "' collection '" + pv.collection + "' descriptor '" + pv.descriptorPath +
                                         "' sceneObject '" + pv.sceneObjectId + "' component " + std::to_string(pv.componentIndex));
        }
        g_scene = std::move(scene);
    } else {
        g_sceneDiag = std::string("BuildCudaHostScene failed: ") + fv::simulation::CudaSceneBuildResultName(r);
    }
    return __real__ZN23ReplaySimulationSession18InstallStaticSceneE26StaticSceneModelCollection(self, models);
}

std::uint32_t __real__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(
        ReplayStaticCorpusCollection *, ReplayCheckpointContactObserver &);
std::uint32_t __wrap__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(
        ReplayStaticCorpusCollection *self, ReplayCheckpointContactObserver &observer) {
    const std::uint32_t n =
            __real__ZN28ReplayStaticCorpusCollection23InstallRaceTriggerHooksER31ReplayCheckpointContactObserver(self, observer);
    BuildCollectionTriggers(self);
    return n;
}

ReplaySimulationRunResult
__real__ZN23ReplaySimulationRuntime5StartERK26ReplaySimulationDefinitionR14ReplayMapSceneRK6GmIso4RK17ReplayControlTickj(
        ReplaySimulationRuntime *, const ReplaySimulationDefinition &, ReplayMapScene &, const GmIso4 &,
        const ReplayControlTick &, unsigned);
ReplaySimulationRunResult
__wrap__ZN23ReplaySimulationRuntime5StartERK26ReplaySimulationDefinitionR14ReplayMapSceneRK6GmIso4RK17ReplayControlTickj(
        ReplaySimulationRuntime *self, const ReplaySimulationDefinition &definition, ReplayMapScene &scene,
        const GmIso4 &spawn, const ReplayControlTick &first, unsigned seed) {
    auto cfg = std::make_unique<fv::simulation::CudaHostStaticConfiguration>();
    if (fv::simulation::BuildCudaHostStaticConfiguration(definition, cfg.get()) ==
        fv::simulation::CudaStaticConfigurationBuildResult::Success)
        g_config = std::move(cfg);
    if (!g_haveStartSpawn) {
        g_startSpawn = spawn;
        g_haveStartSpawn = true;
    }
    return __real__ZN23ReplaySimulationRuntime5StartERK26ReplaySimulationDefinitionR14ReplayMapSceneRK6GmIso4RK17ReplayControlTickj(
            self, definition, scene, spawn, first, seed);
}

void __real__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(
        CHmsCollisionManagerSZone *, CHmsCollisionBuffer &, CHmsCorpus *);
void __wrap__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(
        CHmsCollisionManagerSZone *self, CHmsCollisionBuffer &buffer, CHmsCorpus *corpus) {
    if (!g_fv.taken && corpus != nullptr && corpus->Item() != nullptr) SnapshotZone(self, corpus);
    __real__ZN20CHmsCollisionManager5SZone22DetectCollisionsCorpusER19CHmsCollisionBufferP10CHmsCorpus(self, buffer, corpus);
}
}  // extern "C"

namespace {

struct Totals {
    std::uint64_t replays = 0, tris = 0, bytes = 0;
    double sceneMs = 0, compileMs = 0, openMs = 0;
    std::uint64_t waterProbes = 0, waterMaps = 0;
    std::uint64_t protoExtraTris = 0, protoExtraMaps = 0, movingPairs = 0, movingTris = 0, enclosureTris = 0;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> protoExtra;
    double maxCompileMs = 0, maxBytes = 0;
    std::string maxCompileMap, maxBytesMap;
} g_tot;

std::uint32_t g_probes = 20000;
std::string g_outDir;
bool g_keep = false;  // keep the saved packs (--keep); otherwise each is removed after its round trip

void CheckTriangles(const std::string &name, const w::WorldPackView &v) {
    const std::vector<FvTri> &fvt = g_fv.tris;
    g_stats.Check(g_fv.taken, "fv_records", name + ": no FV detection seen during the load");
    g_stats.Check(g_fv.order == "134", "fv_group_order", name + ": target groups " + g_fv.order);
    std::uint32_t packMoving = 0u;
    for (const w::Actor &a : v.actors)
        if (a.instanceCount && (v.instances[a.firstInstance].flags & w::kInstMovingPair)) ++packMoving;
    g_stats.Check(g_fv.nonMesh == 0u && g_fv.group3Records == 0u, "fv_scope",
                  name + ": non-mesh " + std::to_string(g_fv.nonMesh) + ", group-3 records " + std::to_string(g_fv.group3Records));
    g_stats.Check(g_fv.movingPairs == packMoving, "moving_pair_corpora",
                  name + ": FV " + std::to_string(g_fv.movingPairs) + ", pack " + std::to_string(packMoving));
    g_tot.movingPairs += g_fv.movingPairs;
    if (!g_stats.quiet && (g_fv.movingCorpora || g_fv.skippedNoCollision))
        std::printf("  note %s: FV target groups hold %u non-static corpora (%u enabled pairs), %u records without surface collision\n",
                    name.c_str(), g_fv.movingCorpora, g_fv.movingPairs, g_fv.skippedNoCollision);
    if (!g_stats.quiet)
        for (const std::string &m : g_fv.movingInfo) std::printf("    %s\n", m.c_str());
    g_stats.Check(fvt.size() == v.tris.size(), "tri_count",
                  name + ": FV " + std::to_string(fvt.size()) + ", pack " + std::to_string(v.tris.size()));
    // pack triangles in FV order
    std::vector<std::uint32_t> byOrder(v.tris.size(), UINT32_MAX);
    for (std::size_t i = 0; i < v.tris.size(); ++i) byOrder[v.triInfo[i].fvOrder] = static_cast<std::uint32_t>(i);
    std::unordered_map<const CHmsCorpus *, std::uint32_t> corpusActor;
    std::unordered_map<std::uint32_t, const CHmsCorpus *> actorCorpus;
    std::uint64_t badPos = 0, badMat = 0, badGroup = 0, badPlane = 0, badIso = 0, badModelTri = 0, badRecord = 0, badActor = 0,
                  badTrigger = 0, badPeer = 0;
    std::string first;
    const std::size_t n = std::min(fvt.size(), v.tris.size());
    for (std::size_t k = 0; k < n; ++k) {
        const FvTri &f = fvt[k];
        const std::uint32_t ti = byOrder[k];
        const tw::Tri &t = v.tris[ti];
        const w::TriInfo &x = v.triInfo[ti];
        const w::Instance &in = v.instances[x.instance];
        const w::Model &m = v.models[in.model];
        const w::ModelTri &mt = v.modelTris[m.firstTri + x.modelTri];
        bool pos = true;
        for (int c = 0; c < 3; ++c)
            for (int a = 0; a < 3; ++a) pos = pos && Bits(t.v[c][a]) == f.v[3 * c + a];
        const std::uint32_t mat = t.meta & tw::kMetaMaterialMask;
        const bool trig = (t.meta & tw::kMetaTrigger) != 0u;
        const bool carIgn = (t.meta & tw::kMetaCarIgnores) != 0u;
        std::uint32_t isoW[12];
        std::memcpy(isoW, in.iso, 48);
        if (!pos) ++badPos;
        if (mat != f.material || (mt.meta & w::kModelTriMaterialMask) != f.material || carIgn != (f.material == 13u || f.material == 23u))
            ++badMat;
        if (trig != (f.group == 1u) || x.group != f.group || in.group != f.group) ++badGroup;
        if (Bits(mt.n[0]) != f.n[0] || Bits(mt.n[1]) != f.n[1] || Bits(mt.n[2]) != f.n[2] || Bits(mt.d) != f.d) ++badPlane;
        if (std::memcmp(isoW, f.iso, 48) != 0) ++badIso;
        if (x.modelTri != f.meshTri) ++badModelTri;
        if (in.fvRecord != f.recordLeaf) ++badRecord;
        const std::uint32_t actor = t.meta >> tw::kMetaActorShift;
        auto ca = corpusActor.emplace(f.corpus, actor);
        auto ac = actorCorpus.emplace(actor, f.corpus);
        const auto order = g_corpusOrder.find(f.corpus);
        if (ca.first->second != actor || ac.first->second != f.corpus || order == g_corpusOrder.end() || order->second != actor ||
            in.actor != actor)
            ++badActor;
        // trigger id = rank of the corpus among race-trigger corpora (FV installation order)
        const auto tr = g_corpusTrigger.find(f.corpus);
        const std::uint16_t want = (f.group == 1u && tr != g_corpusTrigger.end()) ? static_cast<std::uint16_t>(tr->second) : w::kNoTrigger;
        if (x.trigger != want || v.actors[actor].trigger != (tr != g_corpusTrigger.end() ? want : w::kNoTrigger)) ++badTrigger;
        if (first.empty() && !(pos && mat == f.material && trig == (f.group == 1u)))
            first = " first at FV index " + std::to_string(k) + " (pack tri " + std::to_string(ti) + ")";
    }
    // per actor: reference booster axis and corpus id
    for (const auto &kv : corpusActor) {
        const CHmsCorpus *c = kv.first;
        const w::Actor &a = v.actors[kv.second];
        GmVec3 dir;
        c->GetLocation()->GetDir(dir);
        if (Bits(a.peerZ[0]) != Bits(dir.x) || Bits(a.peerZ[1]) != Bits(dir.y) || Bits(a.peerZ[2]) != Bits(dir.z) ||
            a.peerId != c->Id().Value())
            ++badPeer;
    }
    g_stats.Check(badPos == 0, "tri_positions", name + ": " + std::to_string(badPos) + " triangles differ" + first);
    g_stats.Check(badMat == 0, "tri_materials", name + ": " + std::to_string(badMat));
    g_stats.Check(badGroup == 0, "tri_groups", name + ": " + std::to_string(badGroup));
    g_stats.Check(badPlane == 0, "tri_stored_planes", name + ": " + std::to_string(badPlane));
    g_stats.Check(badIso == 0, "instance_iso", name + ": " + std::to_string(badIso));
    g_stats.Check(badModelTri == 0, "model_tri_index", name + ": " + std::to_string(badModelTri));
    g_stats.Check(badRecord == 0, "fv_record_order", name + ": " + std::to_string(badRecord));
    g_stats.Check(badActor == 0, "tri_actor", name + ": " + std::to_string(badActor));
    g_stats.Check(badTrigger == 0, "tri_trigger_id", name + ": " + std::to_string(badTrigger));
    g_stats.Check(badPeer == 0, "actor_peer", name + ": " + std::to_string(badPeer) + " of " + std::to_string(corpusActor.size()));

    // set equality independent of order (diagnoses an order-only difference)
    if (badPos != 0 && fvt.size() == v.tris.size()) {
        auto key = [](const std::uint32_t *v9, std::uint32_t mat) {
            std::array<std::uint32_t, 10> k{};
            std::memcpy(k.data(), v9, 36);
            k[9] = mat;
            return k;
        };
        std::vector<std::array<std::uint32_t, 10>> a, b;
        for (const FvTri &f : fvt) a.push_back(key(f.v, f.material));
        for (const tw::Tri &t : v.tris) {
            std::uint32_t v9[9];
            std::memcpy(v9, t.v, 36);
            b.push_back(key(v9, t.meta & tw::kMetaMaterialMask));
        }
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::printf("  note %s: as sets the triangles are %s\n", name.c_str(), a == b ? "EQUAL (order differs)" : "different");
    }
    // model -> world (instance flag) re-check for permutation instances
    std::uint64_t badPerm = 0;
    for (const w::Instance &in : v.instances) {
        if (!(in.flags & w::kInstPermExact)) continue;
        float R[3][3];
        w::RotationFromCode(in.rotation, R);
        const w::Model &m = v.models[in.model];
        for (std::uint32_t k = 0; k < m.triCount; ++k) {
            const w::ModelTri &mt = v.modelTris[m.firstTri + k];
            const tw::Tri &t = v.tris[in.firstTri + k];
            for (int c = 0; c < 3; ++c) {
                const w::Vec3f &lv = v.modelVerts[m.firstVert + mt.v[c]];
                const float l[3] = {lv.x, lv.y, lv.z};
                for (int i = 0; i < 3; ++i) {
                    float p = 0.0f;
                    for (int j = 0; j < 3; ++j)
                        if (R[i][j] != 0.0f) p = R[i][j] * l[j] + in.iso[9 + i];
                    if (Bits(p) != Bits(t.v[c][i])) ++badPerm;
                }
            }
        }
    }
    g_stats.Check(badPerm == 0, "model_to_world", name + ": " + std::to_string(badPerm) + " coordinates");
}

void CheckRace(const std::string &name, const w::WorldPackView &v, const twin_fv::Clone &clone) {
    // trigger table == FV's corpus collection
    bool same = v.triggers.size() == g_collTriggers.size();
    std::string why = "pack " + std::to_string(v.triggers.size()) + ", collection " + std::to_string(g_collTriggers.size());
    for (std::size_t i = 0; same && i < v.triggers.size(); ++i) {
        const TwTrigger &a = v.triggers[i];
        const CollectionTrigger &c = g_collTriggers[i];
        same = a.role == c.t.role && a.currentTransform == c.t.currentTransform && a.slot == c.t.slot &&
               (a.spawn != tw::race::kNoSpawn) == c.hasSpawn;
        if (same && c.hasSpawn) {
            std::uint32_t wc[12];
            PutIsoWords(wc, c.spawn);
            same = a.spawn < v.spawns.size() && std::memcmp(v.spawns[a.spawn].iso, wc, 48) == 0;
        }
        if (!same) why += ", first difference at trigger " + std::to_string(i);
    }
    g_stats.Check(same, "race_map_from_actors", name + ": " + why);
    // spawn table: A's RaceMapStore recipe from FV's collection
    std::vector<std::array<std::uint32_t, 12>> isos;
    std::vector<TwSpawn> ref;
    auto spawnFor = [&](const GmIso4 &iso) {
        std::array<std::uint32_t, 12> wv;
        PutIsoWords(wv.data(), iso);
        for (std::size_t i = 0; i < isos.size(); ++i)
            if (isos[i] == wv) return;
        TwSpawn s{};
        GmQuat q;
        q.Set(iso.rotation);
        s.pos[0] = iso.translation.x;
        s.pos[1] = iso.translation.y;
        s.pos[2] = iso.translation.z;
        s.quat[0] = q.x;
        s.quat[1] = q.y;
        s.quat[2] = q.z;
        s.quat[3] = q.w;
        std::memcpy(s.iso, wv.data(), 48);
        ref.push_back(s);
        isos.push_back(wv);
    };
    const GmIso4 start = clone.race.player.CaptureRuntimeClone().previousSpawnLocation;
    spawnFor(start);
    for (const CollectionTrigger &c : g_collTriggers)
        if (c.hasSpawn) spawnFor(c.spawn);
    bool spawnsSame = ref.size() == v.spawns.size();
    for (std::size_t i = 0; spawnsSame && i < ref.size(); ++i) spawnsSame = std::memcmp(&ref[i], &v.spawns[i], sizeof(TwSpawn)) == 0;
    g_stats.Check(spawnsSame, "spawn_table", name + ": pack " + std::to_string(v.spawns.size()) + ", FV " + std::to_string(ref.size()));
    if (g_haveStartSpawn) {
        std::uint32_t a[12], b[12];
        PutIsoWords(a, start);
        PutIsoWords(b, g_startSpawn);
        g_stats.Check(std::memcmp(a, b, 48) == 0, "start_spawn_is_runtime_start", name);
    }
    const auto &p = clone.race.progress;
    g_stats.Check(v.header->requiredCps == p.requiredCheckpointCount, "required_cps",
                  name + ": pack " + std::to_string(v.header->requiredCps) + ", FV " + std::to_string(p.requiredCheckpointCount));
    g_stats.Check(v.header->requiredLaps == p.requiredLapCount, "required_laps",
                  name + ": pack " + std::to_string(v.header->requiredLaps) + ", FV " + std::to_string(p.requiredLapCount));
    g_stats.Check(p.installedTriggerCount <= v.triggers.size(), "installed_triggers", name);
    g_stats.Check(tw::race::TwRaceMapValid(v.RaceMap()), "race_map_valid", name);
    // trigger volumes cover exactly the trigger triangles
    std::uint64_t volTris = 0;
    for (const w::TriggerVolume &tv : v.triggerVolumes) volTris += tv.triCount;
    g_stats.Check(volTris == v.header->triggerTris, "trigger_volumes", name);
}

void CheckWater(const std::string &name, const w::WorldPackView &v) {
    const w::WaterGrid &g = v.header->water;
    g_stats.Check(g.known == 1u, "water_known", name);
    if (!g.known || g_fv.zone == nullptr) return;
    tw::model::WaterZone z;
    z.enabled = g.enabled;
    z.cellSize[0] = g.cellSize[0];
    z.cellSize[1] = g.cellSize[1];
    z.origin[0] = g.origin[0];
    z.origin[1] = g.origin[1];
    z.dims[0] = g.dims[0];
    z.dims[1] = g.dims[1];
    z.outsideValue = g.outsideValue;
    z.surfaceHeight = g.surfaceHeight;
    z.secondaryCullHeight = g.secondaryCullHeight;
    z.cells = v.waterCells.ptr;
    if (g.enabled) ++g_tot.waterMaps;
    std::mt19937 rng(12345u);
    const float lo[3] = {v.header->boundsLo[0] - 64.0f, v.header->boundsLo[1] - 8.0f, v.header->boundsLo[2] - 64.0f};
    const float hi[3] = {v.header->boundsHi[0] + 64.0f, v.header->boundsHi[1] + 8.0f, v.header->boundsHi[2] + 64.0f};
    std::uniform_real_distribution<float> ux(lo[0], hi[0]), uy(lo[1], hi[1]), uz(lo[2], hi[2]), uh(0.0f, 4.0f);
    std::uint64_t bad = 0, accepted = 0;
    const CSceneVehicleWaterZone &fz = g_fv.zone->waterZone;
    auto probe = [&](float x, float zc, float lowerY, float upperY) {
        const bool a = fz.AcceptsRegion(GmVec2{x, zc}, lowerY, upperY);
        const bool b = z.AcceptsRegion(x, zc, lowerY, upperY);
        if (a != b) ++bad;
        if (a) ++accepted;
        ++g_tot.waterProbes;
    };
    for (std::uint32_t i = 0; i < g_probes; ++i) {
        const float y = (i & 1u) && g.enabled ? g.surfaceHeight - 2.0f + uh(rng) : uy(rng);
        probe(ux(rng), uz(rng), y, y + uh(rng));
    }
    if (g.enabled)  // every cell centre, straddling the surface
        for (std::uint32_t cy = 0; cy < g.dims[1]; ++cy)
            for (std::uint32_t cx = 0; cx < g.dims[0]; ++cx)
                probe(g.origin[0] + (cx + 0.5f) * g.cellSize[0], g.origin[1] + (cy + 0.5f) * g.cellSize[1], g.surfaceHeight - 1.0f,
                      g.surfaceHeight + 1.0f);
    g_stats.Check(bad == 0, "water_probes", name + ": " + std::to_string(bad) + " differ");
    if (!g_stats.quiet && g.enabled)
        std::printf("  water %s: enabled, %ux%u cells, surface %.3f, %llu accepted probes\n", name.c_str(), g.dims[0], g.dims[1],
                    g.surfaceHeight, static_cast<unsigned long long>(accepted));
}

void CheckBenchWorld(const std::string &name, const w::WorldPackView &v) {
    tw::World b, p;
    tw::SourceScene bs, ps;
    std::string diag;
    const bool ok = tw::CompileTwinWorld(*g_scene, &b, &diag, &bs);
    w::MakeBenchWorld(v, &p, &ps);
    // B's mesh-local source (architect C's input): per surface, equal when the surface sets agree
    bool srcSame = bs.surfaces.size() == ps.surfaces.size() && bs.vertices == ps.vertices && bs.tris.size() == ps.tris.size();
    for (std::size_t i = 0; srcSame && i < bs.surfaces.size(); ++i)
        srcSame = std::memcmp(&bs.surfaces[i], &ps.surfaces[i], sizeof(tw::SourceSurface)) == 0;
    srcSame = srcSame && std::memcmp(bs.tris.data(), ps.tris.data(), bs.tris.size() * sizeof(tw::SourceTri)) == 0;
    g_stats.Check(srcSame, "bench_source_eq_prototype",
                  name + ": B " + std::to_string(bs.surfaces.size()) + " surfaces, pack " + std::to_string(ps.surfaces.size()));
    // B's prototype restricted to FV's collision set (the pack's surfaces) must equal the bench world.
    std::vector<std::uint8_t> inPack(g_scene->surfaces.size(), 0u);
    for (const w::Instance &in : v.instances) inPack[in.fvSurface] = 1u;
    std::vector<tw::Tri> bt;
    std::vector<tw::TriExtra> bx;
    std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> extraBySort;  // B-only surfaces: (surfaces, tris)
    std::vector<std::uint8_t> counted(g_scene->surfaces.size(), 0u);
    for (std::size_t i = 0; ok && i < b.tris.size(); ++i) {
        const std::uint32_t si = b.extra[i].surface;
        if (inPack[si]) {
            bt.push_back(b.tris[i]);
            bx.push_back(b.extra[i]);
            continue;
        }
        const auto &s = g_scene->surfaces[si];
        const auto &a = g_scene->actors[s.actorIndex];
        const std::string key = "group " + std::to_string(static_cast<unsigned>(a.itemProperties.collisionGroup)) + " static " +
                                std::to_string(a.itemProperties.collisionStatic ? 1 : 0) + " purpose " + std::to_string(a.purpose) +
                                " checkpoint " + std::to_string(a.hasCheckpoint ? 1 : 0);
        auto &e = extraBySort[key];
        if (!counted[si]) {
            counted[si] = 1u;
            ++e.first;
        }
        ++e.second;
    }
    bool same = ok && bt.size() == p.tris.size();
    same = same && std::memcmp(bt.data(), p.tris.data(), bt.size() * sizeof(tw::Tri)) == 0;
    same = same && std::memcmp(bx.data(), p.extra.data(), bx.size() * sizeof(tw::TriExtra)) == 0;
    g_stats.Check(same, "bench_world_eq_prototype_on_fv_set",
                  name + ": B " + std::to_string(bt.size()) + " tris on FV's set, pack " + std::to_string(p.tris.size()));
    std::uint32_t packOnly = 0u;
    for (std::size_t i = 0; i < p.extra.size(); ++i) {
        const std::uint32_t si = p.extra[i].surface;
        const auto &s = g_scene->surfaces[si];
        const auto &a = g_scene->actors[s.actorIndex];
        const bool bTakes = a.hasCheckpoint || a.purpose == static_cast<std::uint32_t>(StaticScenePurpose::CheckpointTrigger) ||
                            a.itemProperties.collisionGroup == CHmsItem::ECollisionGroup_Static;
        if (!bTakes) ++packOnly;
    }
    g_stats.Check(packOnly == 0u, "pack_subset_of_prototype", name + ": " + std::to_string(packOnly) + " pack triangles B lacks");
    g_tot.protoExtraTris += b.tris.size() - bt.size();
    if (!extraBySort.empty()) ++g_tot.protoExtraMaps;
    for (const auto &kv : extraBySort) {
        auto &t = g_tot.protoExtra[kv.first];
        t.first += kv.second.first;
        t.second += kv.second.second;
        if (!g_stats.quiet)
            std::printf("  note %s: prototype-only surfaces (%s): %u surfaces, %u tris\n", name.c_str(), kv.first.c_str(),
                        kv.second.first, kv.second.second);
    }
}

// --describe: the moving-pair corpora (provenance, per-surface boxes, materials, normal classes)
void DescribeMoving(const w::WorldPackView &v) {
    for (const std::string &p : g_movingProvenance) std::printf("  moving-pair model: %s\n", p.c_str());
    for (const w::Instance &in : v.instances) {
        if (!(in.flags & w::kInstMovingPair)) continue;
        std::uint32_t up = 0, down = 0, side = 0, mats = 0;
        for (std::uint32_t k = 0; k < in.triCount; ++k) {
            const w::TriPlane &pl = v.triPlanes[in.firstTri + k];
            if (pl.n[1] > 0.7f) ++up;
            else if (pl.n[1] < -0.7f) ++down;
            else ++side;
            mats |= 1u << (v.tris[in.firstTri + k].meta & 31u);
        }
        std::printf("    surface %u: %4u tris, box (%.1f %.1f %.1f)-(%.1f %.1f %.1f), normals up %u down %u side %u, materials 0x%x, rot %u\n",
                    in.fvSurface, in.triCount, in.lo[0], in.lo[1], in.lo[2], in.hi[0], in.hi[1], in.hi[2], up, down, side, mats,
                    unsigned(in.rotation));
    }
    // huge instances (extent > 2 km): world enclosure candidates
    for (const w::Instance &in : v.instances) {
        const float ex = std::max(in.hi[0] - in.lo[0], std::max(in.hi[1] - in.lo[1], in.hi[2] - in.lo[2]));
        if (ex < 2000.0f) continue;
        const w::Actor &a = v.actors[in.actor];
        std::uint32_t up = 0, down = 0, side = 0, mats = 0;
        double r2min = 1e300, r2max = 0;
        const float c[3] = {0.5f * (in.lo[0] + in.hi[0]), 0.5f * (in.lo[1] + in.hi[1]), 0.5f * (in.lo[2] + in.hi[2])};
        for (std::uint32_t k = 0; k < in.triCount; ++k) {
            const w::TriPlane &pl = v.triPlanes[in.firstTri + k];
            const tw::Tri &t = v.tris[in.firstTri + k];
            if (pl.n[1] > 0.7f) ++up;
            else if (pl.n[1] < -0.7f) ++down;
            else ++side;
            mats |= 1u << (t.meta & 31u);
            for (int q = 0; q < 3; ++q) {
                double r2 = 0;
                for (int a2 = 0; a2 < 3; ++a2) r2 += double(t.v[q][a2] - c[a2]) * (t.v[q][a2] - c[a2]);
                r2min = std::min(r2min, r2);
                r2max = std::max(r2max, r2);
            }
        }
        std::printf("    HUGE surface %u actor %u (group %u purpose %u static %u): %u tris, box (%.1f %.1f %.1f)-(%.1f %.1f %.1f), "
                    "radius from box centre %.1f-%.1f, normals up %u down %u side %u, materials 0x%x\n",
                    in.fvSurface, in.actor, unsigned(a.group), unsigned(a.purpose), (a.flags & w::kActorCollisionStatic) ? 1u : 0u,
                    in.triCount, in.lo[0], in.lo[1], in.lo[2], in.hi[0], in.hi[1], in.hi[2], std::sqrt(r2min), std::sqrt(r2max), up, down,
                    side, mats);
    }
    for (const std::string &p : g_hugeProvenance) std::printf("  huge model: %s\n", p.c_str());
    std::printf("  map bounds (%.1f %.1f %.1f)-(%.1f %.1f %.1f)\n", v.header->boundsLo[0], v.header->boundsLo[1], v.header->boundsLo[2],
                v.header->boundsHi[0], v.header->boundsHi[1], v.header->boundsHi[2]);
}

// The default pack (enclosure left out) = the full pack minus the enclosure actors' triangles,
// same bits, same relative FV order; and every left-out triangle is > 1 km from the rest.
void CheckEnclosure(const std::string &name, const w::WorldPackView &full, const w::WorldPackView &def) {
    std::vector<std::uint32_t> keep;
    std::uint32_t encl = 0u, enclActors = 0u;
    for (const w::Actor &a : full.actors) enclActors += (a.flags & w::kActorWorldEnclosure) ? 1u : 0u;
    for (std::size_t i = 0; i < full.tris.size(); ++i) {
        if (full.actors[full.tris[i].meta >> tw::kMetaActorShift].flags & w::kActorWorldEnclosure) ++encl;
        else keep.push_back(static_cast<std::uint32_t>(i));
    }
    bool same = keep.size() == def.tris.size() && def.header->enclosureTris == encl &&
                ((def.header->flags & w::kPackEnclosureExcluded) != 0u) == (encl != 0u) && full.header->enclosureTris == 0u;
    for (std::size_t k = 0; same && k < keep.size(); ++k)
        same = std::memcmp(&full.tris[keep[k]], &def.tris[k], sizeof(tw::Tri)) == 0 &&
               full.triInfo[keep[k]].modelTri == def.triInfo[k].modelTri && full.triInfo[keep[k]].trigger == def.triInfo[k].trigger;
    // relative FV order kept: sort kept triangles by full fvOrder; default fvOrder must then be 0, 1, 2, ...
    if (same) {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> o;
        for (std::size_t k = 0; k < keep.size(); ++k) o.push_back({full.triInfo[keep[k]].fvOrder, def.triInfo[k].fvOrder});
        std::sort(o.begin(), o.end());
        for (std::size_t k = 0; same && k < o.size(); ++k) same = o[k].second == k;
    }
    g_stats.Check(same, "default_pack_is_full_minus_enclosure",
                  name + ": full " + std::to_string(full.tris.size()) + ", default " + std::to_string(def.tris.size()) +
                          ", enclosure " + std::to_string(encl));
    g_stats.Check(enclActors == 1u && encl == 960u, "enclosure_found",
                  name + ": " + std::to_string(enclActors) + " actors, " + std::to_string(encl) + " triangles");
    // distance: every enclosure triangle's box is > 1 km from the default pack's bounds
    std::uint32_t near = 0u;
    for (const tw::Tri &t : full.tris) {
        if (!(full.actors[t.meta >> tw::kMetaActorShift].flags & w::kActorWorldEnclosure)) continue;
        bool apart = false;
        for (int a = 0; a < 3; ++a)
            apart = apart || t.lo[a] > def.header->boundsHi[a] + 1000.0f || t.hi[a] < def.header->boundsLo[a] - 1000.0f;
        if (!apart) ++near;
    }
    g_stats.Check(near == 0u, "enclosure_far", name + ": " + std::to_string(near) + " triangles within 1 km of the map");
}

void RunReplay(const std::string &packs, const std::string &path) {
    const std::string name = path.substr(path.find_last_of('/') + 1);
    ResetHooks();
    auto bytes = fv::ReadNativeReplayFile(path, fv::ReplayIdentity{path});
    if (!bytes) {
        g_stats.Check(false, "load", name + ": read");
        return;
    }
    auto source = fv::OpenInstalledPackDirectory(packs);
    if (!source) {
        g_stats.Check(false, "load", "packs");
        return;
    }
    fvx::PhysicsSandboxOptions options;
    options.backend = fv::SimulationBackend::Reference;
    auto created = fvx::CreatePhysicsSandbox(std::move(source).Value(), options);
    if (!created) {
        g_stats.Check(false, "load", name + ": sandbox");
        return;
    }
    fvx::PhysicsSandbox sb = std::move(created).Value();
    auto loaded = sb.LoadReplay({bytes.Value().data(), bytes.Value().size()}, fv::ReplayIdentity{path});
    if (!loaded) {
        std::printf("  skip %s: FV load failed (%s)\n", name.c_str(), loaded.Error().diagnostic.c_str());
        g_stats.Check(true, "fv_load_failed_skipped", name);
        return;
    }
    if (!g_scene) {
        g_stats.Check(false, "load", name + ": " + g_sceneDiag);
        return;
    }
    auto st = sb.CaptureState();
    std::string err;
    const twin_fv::Clone *clonePtr = st ? twin_fv::PeekClone(st.Value(), &err) : nullptr;
    if (clonePtr == nullptr) {
        g_stats.Check(false, "load", name + ": capture " + err);
        return;
    }
    const twin_fv::Clone clone = *clonePtr;
    const GmIso4 start = clone.race.player.CaptureRuntimeClone().previousSpawnLocation;

    tw::fv::WorldCompileInputs in;
    in.scene = g_scene.get();
    in.startSpawn = &start;
    in.requiredLaps = clone.race.progress.requiredLapCount;
    in.config = g_config.get();
    // dpack: the default package (world enclosure left out) - timed, saved, sizes reported.
    // pack:  the same with includeWorldEnclosure, compared with FV's complete collision set.
    w::WorldPack dpack, again, pack;
    std::string diag;
    in.includeWorldEnclosure = false;
    const auto t0 = std::chrono::steady_clock::now();
    const bool compiled = tw::fv::CompileWorldPack(in, &dpack, &diag);
    const auto t1 = std::chrono::steady_clock::now();
    g_stats.Check(compiled, "compile", name + ": " + diag);
    if (!compiled) return;
    w::WorldPackView dv;
    const auto t2 = std::chrono::steady_clock::now();
    const bool opened = w::OpenWorldPack(dpack, &dv, &diag);
    const auto t3 = std::chrono::steady_clock::now();
    g_stats.Check(opened, "open", name + ": " + diag);
    if (!opened) return;
    g_stats.Check(tw::fv::CompileWorldPack(in, &again, &diag) && again.Hash() == dpack.Hash() && again.size() == dpack.size() &&
                          std::memcmp(again.data(), dpack.data(), dpack.size()) == 0,
                  "deterministic", name);
    {   // opt-in: moving-pair corpora ("Flags") left out = the default pack minus exactly those triangles
        tw::fv::WorldCompileInputs nin = in;
        nin.excludeMovingPairCorpora = true;
        w::WorldPack npack;
        w::WorldPackView nv;
        std::uint32_t moving = 0u;
        for (const w::Instance &x : dv.instances)
            if (x.flags & w::kInstMovingPair) moving += x.triCount;
        bool ok = tw::fv::CompileWorldPack(nin, &npack, &diag) && w::OpenWorldPack(npack, &nv, &diag);
        ok = ok && nv.tris.size() + moving == dv.tris.size() && nv.header->excludedMovingPairTris == moving &&
             (nv.header->flags & w::kPackMovingPairsExcluded) != 0u && nv.header->movingPairInstances == 0u;
        std::size_t k = 0;
        for (std::size_t i = 0; ok && i < dv.tris.size(); ++i) {
            if (dv.instances[dv.triInfo[i].instance].flags & w::kInstMovingPair) continue;
            ok = std::memcmp(&dv.tris[i], &nv.tris[k++], sizeof(tw::Tri)) == 0;
        }
        g_stats.Check(ok, "exclude_moving_pairs_option", name + ": " + std::to_string(moving) + " moving-pair triangles");
    }
    in.includeWorldEnclosure = true;
    w::WorldPackView v;
    const bool full = tw::fv::CompileWorldPack(in, &pack, &diag) && w::OpenWorldPack(pack, &v, &diag);
    g_stats.Check(full, "compile_with_enclosure", name + ": " + diag);
    if (!full) return;
    CheckEnclosure(name, v, dv);
    g_stats.Check(v.header->fvSceneHash == g_scene->deterministicHash, "fv_scene_hash", name);
    if (!g_outDir.empty()) {
        const std::string file = g_outDir + "/" + name + ".twpack";
        w::WorldPack back;
        const bool rt = w::SaveWorldPack(file.c_str(), dpack, &diag) && w::LoadWorldPack(file.c_str(), &back, &diag) &&
                        back.size() == dpack.size() && std::memcmp(back.data(), dpack.data(), dpack.size()) == 0;
        g_stats.Check(rt, "save_load", name + ": " + diag);
        if (!g_keep) std::remove(file.c_str());
    }

    if (g_describe) DescribeMoving(v);
    CheckTriangles(name, v);
    CheckRace(name, v, clone);
    CheckWater(name, v);
    CheckBenchWorld(name, v);

    const double compileMs = Ms(t0, t1), openMs = Ms(t2, t3);
    std::uint32_t movingTris = 0u;
    for (const w::Instance &in : v.instances)
        if (in.flags & w::kInstMovingPair) movingTris += in.triCount;
    g_tot.movingTris += movingTris;
    std::uint32_t specialsByMat[32] = {};
    for (const w::SpecialSurface &s : v.specials) specialsByMat[s.material & 31u] += s.triCount;
    std::printf("%-26s tris %7zu (trig %5u, moving %4u, enclosure out %u) inst %5zu models %4zu gen %3u inexact %u actors %5u trig %3zu spawns %3zu cps %2u laps %u "
                "turbo %u/%u fw %u water %s | %6.2f MB | scene %6.1f ms compile %6.1f ms open %5.1f ms\n",
                name.c_str(), dv.tris.size(), v.header->triggerTris, movingTris, dv.header->enclosureTris, v.instances.size(), v.models.size(), v.header->generalInstances,
                v.header->inexactPermInstances, v.header->actorCount, v.triggers.size(), v.spawns.size(), v.header->requiredCps,
                v.header->requiredLaps, specialsByMat[w::kMatTurbo], specialsByMat[w::kMatTurbo2], specialsByMat[w::kMatFreeWheeling],
                v.header->water.enabled ? "on" : "off", dpack.size() / 1048576.0, g_sceneMs, compileMs, openMs);
    ++g_tot.replays;
    g_tot.tris += v.tris.size();
    g_tot.bytes += dpack.size();
    g_tot.enclosureTris += dv.header->enclosureTris;
    g_tot.sceneMs += g_sceneMs;
    g_tot.compileMs += compileMs;
    g_tot.openMs += openMs;
    if (compileMs > g_tot.maxCompileMs) {
        g_tot.maxCompileMs = compileMs;
        g_tot.maxCompileMap = name;
    }
    if (dpack.size() > g_tot.maxBytes) {
        g_tot.maxBytes = static_cast<double>(dpack.size());
        g_tot.maxBytesMap = name;
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s PACKS [--out DIR [--keep]] [--probes N] [--quiet] REPLAY...\n", argv[0]);
        return 2;
    }
    const std::string packs = argv[1];
    std::vector<std::string> replays;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) g_outDir = argv[++i];
        else if (a == "--probes" && i + 1 < argc) g_probes = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        else if (a == "--quiet") g_stats.quiet = true;
        else if (a == "--keep") g_keep = true;
        else if (a == "--describe") g_describe = true;
        else replays.push_back(a);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (const std::string &r : replays) RunReplay(packs, r);
    std::printf("\n%-28s %10s %10s\n", "check", "pass", "fail");
    for (const auto &kv : g_stats.byName)
        std::printf("%-28s %10llu %10llu\n", kv.first.c_str(), static_cast<unsigned long long>(kv.second.first),
                    static_cast<unsigned long long>(kv.second.second));
    if (g_tot.replays)
        std::printf("\n%llu maps, %llu triangles, %.1f MB packs (mean %.2f MB); mean per map: scene export %.1f ms, compile %.1f ms, "
                    "open+hash %.1f ms; %llu water probes on %llu maps with water; wall %.1f s\n",
                    static_cast<unsigned long long>(g_tot.replays), static_cast<unsigned long long>(g_tot.tris), g_tot.bytes / 1048576.0,
                    g_tot.bytes / 1048576.0 / g_tot.replays, g_tot.sceneMs / g_tot.replays, g_tot.compileMs / g_tot.replays,
                    g_tot.openMs / g_tot.replays, static_cast<unsigned long long>(g_tot.waterProbes),
                    static_cast<unsigned long long>(g_tot.waterMaps), Ms(t0, std::chrono::steady_clock::now()) / 1000.0);
    if (g_tot.replays)
        std::printf("max compile %.1f ms (%s), max pack %.2f MB (%s)\n", g_tot.maxCompileMs, g_tot.maxCompileMap.c_str(),
                    g_tot.maxBytes / 1048576.0, g_tot.maxBytesMap.c_str());
    std::printf("FV moving-pair corpora in the car's target groups: %llu (%llu triangles, in the packs; not in FV's CUDA scene)\n",
                static_cast<unsigned long long>(g_tot.movingPairs), static_cast<unsigned long long>(g_tot.movingTris));
    std::printf("world enclosure left out of the default packs: %llu triangles (the full-set checks use includeWorldEnclosure)\n",
                static_cast<unsigned long long>(g_tot.enclosureTris));
    std::printf("B prototype (tw_world_compile.cpp) extra triangles outside FV's collision set: %llu on %llu maps\n",
                static_cast<unsigned long long>(g_tot.protoExtraTris), static_cast<unsigned long long>(g_tot.protoExtraMaps));
    for (const auto &kv : g_tot.protoExtra)
        std::printf("  %s: %llu surfaces, %llu tris\n", kv.first.c_str(), static_cast<unsigned long long>(kv.second.first),
                    static_cast<unsigned long long>(kv.second.second));
    std::printf("%s\n", g_stats.failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
    return g_stats.failures == 0 ? 0 : 1;
}
