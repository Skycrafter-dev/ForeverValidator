#pragma once

#include <cstddef>
#include <vector>

#include "engine/core/engine_types.h"
#include "engine/physics/collision/hms_collision.h"
#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/physics/dynamics/hms_corpus_registry.h"
#include "engine/physics/dynamics/hms_force_field.h"
struct CHmsCorpus;
class OptimizedCpuStaticSurfaceTransformCache;
namespace forevervalidator::simulation {
class OptimizedCpuModel3VehicleForceContext;
}

struct CHmsZone {
    CHmsZone(void);
    virtual ~CHmsZone(void);
    virtual CHmsCollisionManagerSZone *GetCollisionZone(void);
    CSceneVehicleWaterZone *WaterZone(void);
    const CSceneVehicleWaterZone *WaterZone(void) const;
    void CorpusChangeCat(unsigned long indexInCategory,
                         EHmsCorpusCat oldCategory,
                         EHmsCorpusCat newCategory);
    void CorpusChangeBuild(CHmsCorpus *corpus, int enabled);
    void CorpusChangeLightEmitter(CHmsCorpus *corpus, int enabled);
    void AddField(CHmsForceField *field);
    void RemoveField(CHmsForceField *field);
    virtual void AddCorpus(CHmsCorpus *corpus);
    virtual void RemoveCorpus(CHmsCorpus *corpus);
    void ChangeCorpusCategory(CHmsCorpus &corpus,
                              EHmsCorpusCat newCategory);
    bool AppendWaterPlaneEq(const GmVec4 &plane);
    std::size_t WaterPlaneEqCount(void) const;
    const GmVec4 *WaterPlaneEqAt(std::size_t index) const;

protected:
    bool ContainsCorpus(const CHmsCorpus &corpus) const {
        return corpusesByCategory_.Find(corpus).has_value();
    }
    CHmsCorpus *FirstCorpusOrNull(void) const {
        return corpusesByCategory_.FirstOrNull();
    }
    bool HasForceFields(void) const {
        return !forceFields_.empty();
    }
    CHmsForceField *LastForceField(void) const {
        return forceFields_.empty() ? nullptr : forceFields_.back();
    }
    const std::vector<CHmsForceField *> &ForceFields(void) const {
        return forceFields_;
    }

private:
    CHmsCorpusCategoryStore corpusesByCategory_;
    std::vector<CHmsCorpus *> buildCorpuses_;
    std::vector<CHmsCorpus *> lightEmitterCorpuses_;
    std::vector<CHmsForceField *> forceFields_;
    std::vector<GmVec4> waterPlaneEqs_;
};

struct CHmsZoneDynamic : CHmsZone {
    CHmsZoneDynamic(void);
    ~CHmsZoneDynamic(void) override;
    CHmsCollisionManagerSZone *GetCollisionZone(void) override;
    void BindCollisionManager(CHmsCollisionManagerSZone &managerZone);
    void UnbindCollisionManager(
            const CHmsCollisionManagerSZone &managerZone);
    void ResetSimulationState(void);
    void SetCollisionManagerZone(CHmsCollisionManagerSZone &managerZone);
    void ResetForceFields(float linearDamping, float angularDamping);
    void ComputeCorpusForces(
            CHmsCorpus *corpus,
            float dt);
    void ComputeCorpusForcesOptimizedCpuModel3(
            CHmsCorpus *corpus,
            float dt,
            forevervalidator::simulation::
                    OptimizedCpuModel3VehicleForceContext &context);
    void AddCorpus(CHmsCorpus *corpus) override;
    void RemoveCorpus(CHmsCorpus *corpus) override;
    void SolveImpulse(
            const SHmsPhysicalCollision &collision,
            CHmsPhysicalContact *contactA,
            CHmsPhysicalContact *contactB);
    void ComputeCollisionResponse(
            void);
    void PhysicsStep2(
            void);
    void PhysicsStep2OptimizedCpu(
            void);
    void PhysicsStep2OptimizedCpuNativeBinary32(
            forevervalidator::simulation::
                    OptimizedCpuModel3VehicleForceContext &context);
    void PhysicsStep2OptimizedCpuCached(
            const OptimizedCpuStaticSurfaceTransformCache &transforms);
    void PhysicsStep2OptimizedCpuNativeBinary32Cached(
            const OptimizedCpuStaticSurfaceTransformCache &transforms,
            forevervalidator::simulation::
                    OptimizedCpuModel3VehicleForceContext &context);

private:
    void PhysicsStep2OptimizedCpuCachedImpl(
            const OptimizedCpuStaticSurfaceTransformCache &transforms,
            bool nativeBinary32,
            forevervalidator::simulation::
                    OptimizedCpuModel3VehicleForceContext *model3Context);

    float linearDampingCoef_ = 1.0f;
    float angularDampingCoef_ = 1.0f;
    std::vector<CHmsCorpus *> dynamicCorpuses_;
    std::vector<CHmsCorpus *> zombieCorpuses_;
    CHmsCollisionBuffer collisionBuffer_;
    CHmsCollisionManagerSZone *collisionManagerZone_ = nullptr;
};
