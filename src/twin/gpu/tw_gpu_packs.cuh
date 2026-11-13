// Twin GPU engine: the device packs' host-side owners (tw_gpu.h CarPacks, CollisionPacks).
#ifndef TW_GPU_PACKS_CUH
#define TW_GPU_PACKS_CUH

#include "car_force_consts.h"
#include "tw_gpu.h"
#include "tw_gpu_collide.cuh"
#include "tw_gpu_util.cuh"
#include "tw_race.h"

namespace tw::gpu {

struct CarPacks::Dev {
    DevBuffer car, water, waterCells, race, triggers, spawns;
    const model::CarForceConsts *k = nullptr;
    const model::WaterZone *w = nullptr;   // null: no water on the map
    const race::TwRaceMap *r = nullptr;
};

struct CollisionPacks::Dev {
    DevBuffer qnodes, rankBox, rank, plane, iso, verts, outRec, stats;
    dev::DevWorld w{};
    std::size_t bytes = 0;
};

}  // namespace tw::gpu

#endif
