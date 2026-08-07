#include "simulation/backends/cuda/cuda_collision.cuh"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

namespace collision =
        forevervalidator::simulation::cuda::collision;
using forevervalidator::simulation::CudaPackedSceneHeader;
using forevervalidator::simulation::CudaSceneAccelerationCell;
using forevervalidator::simulation::CudaSceneOctreeCell;
using forevervalidator::simulation::CudaSceneSurface;

struct SceneFixture {
    CudaPackedSceneHeader header{};
    std::vector<CudaSceneAccelerationCell> acceleration;
};

struct ProbeOutput {
    collision::detail::EmptyAirProbeResult result =
            collision::detail::EmptyAirProbeResult::Ineligible;
    GmBoxAligned bounds[2]{};
    collision::CudaHotPathCounters counters{};
};

struct ProbeRun {
    bool executed = false;
    ProbeOutput output{};
};

bool CheckCuda(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

__global__ void RunProbeKernel(
        const CudaPackedSceneHeader *scene,
        const CudaSceneAccelerationCell *acceleration,
        const CudaSceneSurface *surfaces,
        const CudaSceneOctreeCell *octreeCells,
        GmBoxAligned *bounds,
        std::uint32_t shapeCount,
        GmVec3 linearSpeed,
        GmVec3 shortTravel,
        ProbeOutput *output) {
    collision::CudaCollisionSearchScratch scratch{};
    scratch.movingBoundsStorage = bounds;
    scratch.slot = 0u;
    scratch.stride = 1u;
    scratch.shapeCapacity = shapeCount;
    collision::CudaHotPathCounters counters{};
    output->result =
            collision::detail::TryExtendEmptyAirCertificate<true>(
                    scene, acceleration, surfaces, octreeCells,
                    linearSpeed, shortTravel, shapeCount, scratch,
                    &counters);
    for (std::uint32_t index = 0u; index < shapeCount; ++index) {
        output->bounds[index] = bounds[index];
    }
    output->counters = counters;
}

__global__ void CheckBoundaryKernel(bool *touches, bool *separated) {
    const GmBoxAligned left = {
            {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    const GmBoxAligned touching = {
            {2.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    const GmBoxAligned apart = {
            {2.001f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    *touches = collision::detail::BoundsIntersect(left, touching);
    *separated = collision::detail::BoundsIntersect(left, apart);
}

ProbeRun RunProbe(
        const SceneFixture &fixture,
        const std::vector<GmBoxAligned> &bounds,
        GmVec3 linearSpeed = {10.0f, 0.0f, 0.0f},
        GmVec3 shortTravel = {0.1f, 0.0f, 0.0f},
        const std::vector<CudaSceneSurface> &surfaces = {},
        const std::vector<CudaSceneOctreeCell> &octreeCells = {}) {
    ProbeRun run;
    CudaPackedSceneHeader *deviceScene = nullptr;
    CudaSceneAccelerationCell *deviceAcceleration = nullptr;
    CudaSceneSurface *deviceSurfaces = nullptr;
    CudaSceneOctreeCell *deviceOctreeCells = nullptr;
    GmBoxAligned *deviceBounds = nullptr;
    ProbeOutput *deviceOutput = nullptr;

    bool ready = CheckCuda(
            cudaMalloc(&deviceScene, sizeof(*deviceScene)),
            "cudaMalloc(scene)");
    ready = ready && CheckCuda(
            cudaMalloc(
                    &deviceAcceleration,
                    fixture.acceleration.size() *
                            sizeof(fixture.acceleration[0])),
            "cudaMalloc(acceleration)");
    if (!surfaces.empty()) {
        ready = ready && CheckCuda(
                cudaMalloc(
                        &deviceSurfaces,
                        surfaces.size() * sizeof(surfaces[0])),
                "cudaMalloc(surfaces)");
    }
    if (!octreeCells.empty()) {
        ready = ready && CheckCuda(
                cudaMalloc(
                        &deviceOctreeCells,
                        octreeCells.size() * sizeof(octreeCells[0])),
                "cudaMalloc(octree)");
    }
    ready = ready && CheckCuda(
            cudaMalloc(
                    &deviceBounds,
                    bounds.size() * sizeof(bounds[0])),
            "cudaMalloc(bounds)");
    ready = ready && CheckCuda(
            cudaMalloc(&deviceOutput, sizeof(*deviceOutput)),
            "cudaMalloc(output)");

    if (ready) {
        ready = CheckCuda(
                cudaMemcpy(
                        deviceScene, &fixture.header,
                        sizeof(fixture.header), cudaMemcpyHostToDevice),
                "cudaMemcpy(scene)") &&
                CheckCuda(
                        cudaMemcpy(
                                deviceAcceleration,
                                fixture.acceleration.data(),
                                fixture.acceleration.size() *
                                        sizeof(fixture.acceleration[0]),
                                cudaMemcpyHostToDevice),
                        "cudaMemcpy(acceleration)") &&
                CheckCuda(
                        cudaMemcpy(
                                deviceBounds, bounds.data(),
                                bounds.size() * sizeof(bounds[0]),
                                cudaMemcpyHostToDevice),
                        "cudaMemcpy(bounds)");
    }
    if (ready && !surfaces.empty()) {
        ready = CheckCuda(
                cudaMemcpy(
                        deviceSurfaces, surfaces.data(),
                        surfaces.size() * sizeof(surfaces[0]),
                        cudaMemcpyHostToDevice),
                "cudaMemcpy(surfaces)");
    }
    if (ready && !octreeCells.empty()) {
        ready = CheckCuda(
                cudaMemcpy(
                        deviceOctreeCells, octreeCells.data(),
                        octreeCells.size() * sizeof(octreeCells[0]),
                        cudaMemcpyHostToDevice),
                "cudaMemcpy(octree)");
    }
    if (ready) {
        RunProbeKernel<<<1, 1>>>(
                deviceScene, deviceAcceleration, deviceSurfaces,
                deviceOctreeCells, deviceBounds,
                static_cast<std::uint32_t>(bounds.size()),
                linearSpeed, shortTravel, deviceOutput);
        ready = CheckCuda(cudaGetLastError(), "probe launch") &&
                CheckCuda(
                        cudaDeviceSynchronize(), "probe synchronize") &&
                CheckCuda(
                        cudaMemcpy(
                                &run.output, deviceOutput,
                                sizeof(run.output),
                                cudaMemcpyDeviceToHost),
                        "cudaMemcpy(output)");
    }
    run.executed = ready;

    cudaFree(deviceOutput);
    cudaFree(deviceBounds);
    cudaFree(deviceOctreeCells);
    cudaFree(deviceSurfaces);
    cudaFree(deviceAcceleration);
    cudaFree(deviceScene);
    return run;
}

CudaSceneAccelerationCell EmptyRoot() {
    CudaSceneAccelerationCell root{};
    root.subtreeEntryCount = 1u;
    root.surfaceIndex = UINT32_MAX;
    return root;
}

SceneFixture SceneWithGroup(
        const std::vector<CudaSceneAccelerationCell> &cells = {},
        std::uint32_t populatedGroup = 0u) {
    SceneFixture fixture;
    for (std::uint32_t group = 1u; group <= 5u; ++group) {
        auto &range = fixture.header.accelerationGroups[group - 1u];
        range.firstCell = static_cast<std::uint32_t>(
                fixture.acceleration.size());
        if (group == populatedGroup && !cells.empty()) {
            fixture.acceleration.insert(
                    fixture.acceleration.end(), cells.begin(), cells.end());
            range.cellCount = static_cast<std::uint32_t>(cells.size());
        } else {
            fixture.acceleration.push_back(EmptyRoot());
            range.cellCount = 1u;
        }
    }
    fixture.header.accelerationCells.count =
            static_cast<std::uint32_t>(fixture.acceleration.size());
    return fixture;
}

GmBoxAligned ShortBounds(float y = 0.0f) {
    // One-tick cache for a unit live box travelling 0.1 m on X.
    return {{0.05f, y, 0.0f}, {1.1125f, 1.0625f, 1.0625f}};
}

bool EqualBits(const GmBoxAligned &left, const GmBoxAligned &right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

std::vector<CudaSceneAccelerationCell> Tree(GmBoxAligned bounds) {
    return {
            {bounds, 2u, UINT32_MAX},
            {bounds, 1u, 0u},
    };
}

CudaSceneSurface MeshSurface() {
    CudaSceneSurface surface{};
    surface.type = static_cast<std::uint32_t>(
            GmSurf::EGmSurfType::Mesh);
    surface.worldToLocal.rotation.basisX = {1.0f, 0.0f, 0.0f};
    surface.worldToLocal.rotation.basisY = {0.0f, 1.0f, 0.0f};
    surface.worldToLocal.rotation.basisZ = {0.0f, 0.0f, 1.0f};
    surface.octreeCellCount = 2u;
    surface.triangleCount = 1u;
    return surface;
}

std::vector<CudaSceneOctreeCell> Octree(GmBoxAligned bounds) {
    return {
            {bounds, 2u, 0u, 0u},
            {bounds, 1u, 1u, 0u},
    };
}

bool CheckClearCorridor() {
    const std::vector<GmBoxAligned> initial = {
            ShortBounds(-1.5f), ShortBounds(1.5f)};
    const ProbeRun run = RunProbe(SceneWithGroup(), initial);
    if (!run.executed ||
        run.output.result !=
                collision::detail::EmptyAirProbeResult::Clear) {
        std::cerr << "clear corridor did not certify\n";
        return false;
    }
    for (const GmBoxAligned &bounds : run.output.bounds) {
        const float lower = bounds.center.x - bounds.halfExtents.x;
        const float upper = bounds.center.x + bounds.halfExtents.x;
        if (lower > -1.0f || upper < 6.0f) {
            std::cerr << "certificate did not cover five metres\n";
            return false;
        }
    }
    return run.output.counters.
                   emptyAirProbeAccelerationCellVisitCount == 0u;
}

bool CheckFarBranchSkip() {
    const GmBoxAligned far = {
            {100.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    const auto cells = Tree(far);
    const ProbeRun run = RunProbe(
            SceneWithGroup(cells, 1u), {ShortBounds()});
    return run.executed &&
            run.output.result ==
                    collision::detail::EmptyAirProbeResult::Clear &&
            run.output.counters.
                    emptyAirProbeAccelerationCellVisitCount == 1u;
}

bool CheckBlockingGroups() {
    const GmBoxAligned blocking = {
            {2.5f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    const auto cells = Tree(blocking);
    const GmBoxAligned initial = ShortBounds();
    CudaSceneSurface unsupported{};
    unsupported.type = static_cast<std::uint32_t>(
            GmSurf::EGmSurfType::Sphere);
    for (const std::uint32_t group : {1u, 3u, 4u}) {
        SceneFixture fixture = SceneWithGroup(cells, group);
        fixture.header.surfaces.count = 1u;
        const ProbeRun run = RunProbe(
                fixture, {initial}, {10.0f, 0.0f, 0.0f},
                {0.1f, 0.0f, 0.0f}, {unsupported});
        if (!run.executed ||
            run.output.result !=
                    collision::detail::EmptyAirProbeResult::Blocked ||
            !EqualBits(run.output.bounds[0], initial)) {
            std::cerr << "blocking group " << group
                      << " was not rejected conservatively\n";
            return false;
        }
    }
    for (const std::uint32_t group : {2u, 5u}) {
        SceneFixture fixture = SceneWithGroup(cells, group);
        fixture.header.surfaces.count = 1u;
        const ProbeRun run = RunProbe(
                fixture, {initial}, {10.0f, 0.0f, 0.0f},
                {0.1f, 0.0f, 0.0f}, {unsupported});
        if (!run.executed ||
            run.output.result !=
                    collision::detail::EmptyAirProbeResult::Clear) {
            std::cerr << "non-collision group " << group
                      << " unexpectedly blocked the certificate\n";
            return false;
        }
    }
    return true;
}

bool CheckMeshOctreeProof() {
    const GmBoxAligned accelerationBounds = {
            {2.5f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f}};
    const auto acceleration = Tree(accelerationBounds);
    SceneFixture fixture = SceneWithGroup(acceleration, 1u);
    fixture.header.surfaces.count = 1u;
    fixture.header.octreeCells.count = 2u;
    fixture.header.triangles.count = 1u;
    const CudaSceneSurface surface = MeshSurface();

    const GmBoxAligned far = {
            {100.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    const ProbeRun clear = RunProbe(
            fixture, {ShortBounds()}, {10.0f, 0.0f, 0.0f},
            {0.1f, 0.0f, 0.0f}, {surface}, Octree(far));
    if (!clear.executed ||
        clear.output.result !=
                collision::detail::EmptyAirProbeResult::Clear ||
        clear.output.counters.
                        emptyAirProbeOctreeCellVisitCount != 1u) {
        std::cerr << "clear mesh corridor did not certify\n";
        return false;
    }

    const GmBoxAligned blocking = {
            {2.5f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
    const GmBoxAligned initial = ShortBounds();
    const ProbeRun blocked = RunProbe(
            fixture, {initial}, {10.0f, 0.0f, 0.0f},
            {0.1f, 0.0f, 0.0f}, {surface}, Octree(blocking));
    if (!blocked.executed ||
        blocked.output.result !=
                collision::detail::EmptyAirProbeResult::Blocked ||
        !EqualBits(blocked.output.bounds[0], initial) ||
        blocked.output.counters.
                        emptyAirProbeOctreeCellVisitCount != 2u) {
        std::cerr << "intersecting triangle leaf did not block\n";
        return false;
    }

    auto malformedOctree = Octree(blocking);
    malformedOctree[0].subtreeEntryCount = 3u;
    const ProbeRun malformed = RunProbe(
            fixture, {initial}, {10.0f, 0.0f, 0.0f},
            {0.1f, 0.0f, 0.0f}, {surface}, malformedOctree);
    return malformed.executed &&
            malformed.output.result ==
                    collision::detail::EmptyAirProbeResult::Blocked &&
            EqualBits(malformed.output.bounds[0], initial);
}

bool CheckMalformedRangesFailClosed() {
    const GmBoxAligned bounds = {
            {0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f}};
    auto cells = Tree(bounds);
    cells[0].subtreeEntryCount = 3u;
    const GmBoxAligned initial = ShortBounds();
    const ProbeRun malformedTree = RunProbe(
            SceneWithGroup(cells, 1u), {initial});
    if (!malformedTree.executed ||
        malformedTree.output.result !=
                collision::detail::EmptyAirProbeResult::Blocked ||
        !EqualBits(malformedTree.output.bounds[0], initial)) {
        std::cerr << "malformed acceleration tree did not fail closed\n";
        return false;
    }

    SceneFixture zeroCount = SceneWithGroup();
    zeroCount.header.accelerationGroups[0].cellCount = 0u;
    SceneFixture outOfRange = SceneWithGroup();
    outOfRange.header.accelerationGroups[0] = {
            outOfRange.header.accelerationCells.count, 1u};
    SceneFixture oneCellLeaf = SceneWithGroup();
    oneCellLeaf.acceleration[
            oneCellLeaf.header.accelerationGroups[0].firstCell].
                    surfaceIndex = 0u;
    oneCellLeaf.header.surfaces.count = 1u;
    CudaSceneSurface surface{};
    for (const ProbeRun run : {
             RunProbe(zeroCount, {initial}),
             RunProbe(outOfRange, {initial}),
             RunProbe(
                     oneCellLeaf, {initial},
                     {10.0f, 0.0f, 0.0f},
                     {0.1f, 0.0f, 0.0f}, {surface})}) {
        if (!run.executed ||
            run.output.result !=
                    collision::detail::EmptyAirProbeResult::Blocked ||
            !EqualBits(run.output.bounds[0], initial)) {
            std::cerr << "malformed zero/one-cell range escaped proof\n";
            return false;
        }
    }
    return true;
}

bool CheckAlreadyLongShortCacheIsRetained() {
    const GmBoxAligned alreadyLong = {
            {3.0f, 0.0f, 0.0f}, {4.0625f, 1.0625f, 1.0625f}};
    const ProbeRun run = RunProbe(
            SceneWithGroup(), {alreadyLong},
            {10.0f, 0.0f, 0.0f}, {6.0f, 0.0f, 0.0f});
    if (!run.executed ||
        run.output.result !=
                collision::detail::EmptyAirProbeResult::Clear ||
        !EqualBits(run.output.bounds[0], alreadyLong)) {
        std::cerr << "already-long short certificate was rewritten\n";
        return false;
    }
    return true;
}

bool CheckInvalidMotionIsIneligible() {
    const GmBoxAligned initial = ShortBounds();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    for (const GmVec3 speed : {
             GmVec3{0.0f, 0.0f, 0.0f},
             GmVec3{nan, 0.0f, 0.0f},
             GmVec3{infinity, 0.0f, 0.0f}}) {
        const ProbeRun run = RunProbe(
                SceneWithGroup(), {initial}, speed);
        if (!run.executed ||
            run.output.result !=
                    collision::detail::EmptyAirProbeResult::Ineligible ||
            !EqualBits(run.output.bounds[0], initial)) {
            std::cerr << "invalid velocity was accepted\n";
            return false;
        }
    }
    const ProbeRun badTravel = RunProbe(
            SceneWithGroup(), {initial}, {10.0f, 0.0f, 0.0f},
            {nan, 0.0f, 0.0f});
    return badTravel.executed &&
            badTravel.output.result ==
                    collision::detail::EmptyAirProbeResult::Ineligible &&
            EqualBits(badTravel.output.bounds[0], initial);
}

bool CheckTouchingBoundsBlock() {
    bool *deviceTouches = nullptr;
    bool *deviceSeparated = nullptr;
    if (!CheckCuda(
                cudaMalloc(&deviceTouches, sizeof(*deviceTouches)),
                "cudaMalloc(touches)") ||
        !CheckCuda(
                cudaMalloc(&deviceSeparated, sizeof(*deviceSeparated)),
                "cudaMalloc(separated)")) {
        cudaFree(deviceSeparated);
        cudaFree(deviceTouches);
        return false;
    }
    CheckBoundaryKernel<<<1, 1>>>(deviceTouches, deviceSeparated);
    bool touches = false;
    bool separated = true;
    const bool ok =
            CheckCuda(cudaGetLastError(), "boundary launch") &&
            CheckCuda(cudaDeviceSynchronize(), "boundary synchronize") &&
            CheckCuda(
                    cudaMemcpy(
                            &touches, deviceTouches, sizeof(touches),
                            cudaMemcpyDeviceToHost),
                    "cudaMemcpy(touches)") &&
            CheckCuda(
                    cudaMemcpy(
                            &separated, deviceSeparated,
                            sizeof(separated), cudaMemcpyDeviceToHost),
                    "cudaMemcpy(separated)");
    cudaFree(deviceSeparated);
    cudaFree(deviceTouches);
    if (!ok || !touches || separated) {
        std::cerr << "touching AABB boundary was not conservative\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        static_cast<void>(cudaGetLastError());
        return 77;
    }
    if (!CheckClearCorridor() ||
        !CheckFarBranchSkip() ||
        !CheckBlockingGroups() ||
        !CheckMeshOctreeProof() ||
        !CheckMalformedRangesFailClosed() ||
        !CheckAlreadyLongShortCacheIsRetained() ||
        !CheckInvalidMotionIsIneligible() ||
        !CheckTouchingBoundsBlock()) {
        return 1;
    }
    return 0;
}
