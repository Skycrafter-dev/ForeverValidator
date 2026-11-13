// Twin GPU engine: device packs and the car-model stage checks (tw_gpu.h).
#include "tw_gpu.h"

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

#include "car_force_consts.h"
#include "response.h"
#include "tw_race.h"
#include "tw_gpu_packs.cuh"
#include "tw_gpu_stages.cuh"
#include "tw_gpu_util.cuh"

namespace tw::gpu {

CarPacks::CarPacks() = default;
CarPacks::~CarPacks() = default;

bool CarPacks::Upload(const model::CarForceConsts &car, const model::WaterZone *water, const race::TwRaceMap &race,
                      std::string *error) {
    auto d = std::make_unique<Dev>();
    if (!d->car.Put(&car, sizeof car, error)) return false;
    d->k = static_cast<const model::CarForceConsts *>(d->car.ptr);
    if (water != nullptr && water->enabled) {
        model::WaterZone w = *water;
        const std::size_t cells = std::size_t(w.dims[0]) * w.dims[1] + 4u;   // + the >= 3 bytes of padding ValueAt may read
        std::vector<std::uint8_t> copy(cells, 0u);
        if (w.cells != nullptr) std::memcpy(copy.data(), w.cells, cells - 4u);
        if (!d->waterCells.Put(copy.data(), copy.size(), error)) return false;
        w.cells = static_cast<const std::uint8_t *>(d->waterCells.ptr);
        if (!d->water.Put(&w, sizeof w, error)) return false;
        d->w = static_cast<const model::WaterZone *>(d->water.ptr);
    }
    race::TwRaceMap r = race;
    if (!d->triggers.Put(race.triggers, sizeof(race::TwTrigger) * race.triggerCount, error)) return false;
    if (!d->spawns.Put(race.spawns, sizeof(race::TwSpawn) * race.spawnCount, error)) return false;
    r.triggers = static_cast<const race::TwTrigger *>(d->triggers.ptr);
    r.spawns = static_cast<const race::TwSpawn *>(d->spawns.ptr);
    if (!d->race.Put(&r, sizeof r, error)) return false;
    d->r = static_cast<const race::TwRaceMap *>(d->race.ptr);
    dev_ = std::move(d);
    return true;
}

namespace {

__global__ void ForcesKernel(const model::CarForceConsts *k, const model::WaterZone *water, const ForcesCase *cases,
                             ForcesResult *out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    model::TwinState s = cases[i].s;
    model::CarForceOut f;
    dev::ForcesIntegrate(*k, water, s, cases[i].in, cases[i].h, f);
    out[i].s = s;
    out[i].f = f;
}

__global__ void RespondKernel(const model::CarForceConsts *k, const RespondCase *cases, const model::RespContact *contacts,
                              RespondResult *out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const RespondCase &c = cases[i];
    model::TwinState s = c.s;
    model::RespTick tick = c.tick;
    model::RespOut o;
    dev::Respond(*k, s, c.R, c.Rw, contacts + c.first, static_cast<int>(c.n), tick, o);
    out[i].s = s;
    out[i].tick = tick;
    out[i].out = o;
}

__global__ void TickBeginKernel(const model::CarForceConsts *k, const race::TwRaceMap *race, const TickBeginCase *cases,
                                RaceResult *out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    model::TwinState s = cases[i].s;
    race::TwRaceEvents ev;
    race::TwRaceEventsClear(ev);
    race::TwRaceTickBegin(*race, s, cases[i].control, cases[i].respawnCount, s.tickMs + 10u, k->restAbsorb, ev);
    out[i].s = s;
    out[i].ev = ev;
}

__global__ void RaceHitsKernel(const race::TwRaceMap *race, const RaceHitsCase *cases, RaceResult *out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    model::TwinState s = cases[i].s;
    race::TwRaceEvents ev = cases[i].ev;
    race::TwRaceOnTriggerHits(*race, s, cases[i].hits, cases[i].count, ev);
    out[i].s = s;
    out[i].ev = ev;
}

constexpr std::uint32_t kThreads = 128u;
std::uint32_t Blocks(std::size_t n) { return static_cast<std::uint32_t>((n + kThreads - 1u) / kThreads); }

bool Finish(const char *what, std::string *error) {
    return Check(cudaGetLastError(), what, error) && Check(cudaDeviceSynchronize(), what, error);
}

}  // namespace

bool RunRespond(const CarPacks &packs, const RespondCase *cases, std::size_t n, const model::RespContact *contacts,
                std::size_t contactCount, RespondResult *out, std::string *error) {
    if (packs.dev() == nullptr) return *error = "RunRespond: packs not uploaded", false;
    if (n == 0u) return true;
    DevBuffer in, ct, res;
    if (!in.Put(cases, sizeof(RespondCase) * n, error) || !ct.Put(contacts, sizeof(model::RespContact) * contactCount, error) ||
        !res.Alloc(sizeof(RespondResult) * n, error))
        return false;
    RespondKernel<<<Blocks(n), kThreads>>>(packs.dev()->k, static_cast<const RespondCase *>(in.ptr),
                                           static_cast<const model::RespContact *>(ct.ptr), static_cast<RespondResult *>(res.ptr),
                                           static_cast<std::uint32_t>(n));
    return Finish("RespondKernel", error) && res.Get(out, sizeof(RespondResult) * n, error);
}

bool RunTickBegin(const CarPacks &packs, const TickBeginCase *cases, std::size_t n, RaceResult *out, std::string *error) {
    if (packs.dev() == nullptr) return *error = "RunTickBegin: packs not uploaded", false;
    if (n == 0u) return true;
    DevBuffer in, res;
    if (!in.Put(cases, sizeof(TickBeginCase) * n, error) || !res.Alloc(sizeof(RaceResult) * n, error)) return false;
    TickBeginKernel<<<Blocks(n), kThreads>>>(packs.dev()->k, packs.dev()->r, static_cast<const TickBeginCase *>(in.ptr),
                                             static_cast<RaceResult *>(res.ptr), static_cast<std::uint32_t>(n));
    return Finish("TickBeginKernel", error) && res.Get(out, sizeof(RaceResult) * n, error);
}

bool RunRaceHits(const CarPacks &packs, const RaceHitsCase *cases, std::size_t n, RaceResult *out, std::string *error) {
    if (packs.dev() == nullptr) return *error = "RunRaceHits: packs not uploaded", false;
    if (n == 0u) return true;
    DevBuffer in, res;
    if (!in.Put(cases, sizeof(RaceHitsCase) * n, error) || !res.Alloc(sizeof(RaceResult) * n, error)) return false;
    RaceHitsKernel<<<Blocks(n), kThreads>>>(packs.dev()->r, static_cast<const RaceHitsCase *>(in.ptr), static_cast<RaceResult *>(res.ptr),
                                            static_cast<std::uint32_t>(n));
    return Finish("RaceHitsKernel", error) && res.Get(out, sizeof(RaceResult) * n, error);
}

bool RunForces(const CarPacks &packs, const ForcesCase *cases, std::size_t n, ForcesResult *out, std::string *error) {
    if (packs.dev() == nullptr) return *error = "RunForces: packs not uploaded", false;
    if (n == 0u) return true;
    DevBuffer in, res;
    if (!in.Put(cases, sizeof(ForcesCase) * n, error) || !res.Alloc(sizeof(ForcesResult) * n, error)) return false;
    const std::uint32_t threads = 128u, blocks = static_cast<std::uint32_t>((n + threads - 1u) / threads);
    ForcesKernel<<<blocks, threads>>>(packs.dev()->k, packs.dev()->w, static_cast<const ForcesCase *>(in.ptr),
                                      static_cast<ForcesResult *>(res.ptr), static_cast<std::uint32_t>(n));
    if (!Check(cudaGetLastError(), "ForcesKernel launch", error) || !Check(cudaDeviceSynchronize(), "ForcesKernel", error))
        return false;
    return res.Get(out, sizeof(ForcesResult) * n, error);
}

bool DeviceInfo(std::string *info) {
    int count = 0;
    if (!Check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", info)) return false;
    if (count == 0) return *info = "no CUDA device", false;
    cudaDeviceProp p{};
    if (!Check(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties", info)) return false;
    *info = std::string(p.name) + ", compute " + std::to_string(p.major) + "." + std::to_string(p.minor) + ", " +
            std::to_string(p.multiProcessorCount) + " SMs, " + std::to_string(p.totalGlobalMem >> 20) + " MiB";
    return true;
}

}  // namespace tw::gpu
