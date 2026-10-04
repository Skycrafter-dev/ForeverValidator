#include "simulation/backends/cuda/cuda_custom_volume.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace forevervalidator::simulation;

int main() {
    try {
        CudaSearchEvaluatorConfiguration volume;
        volume.optionFlags = 0u;
        volume.values[3] = 2.0;
        volume.customVolumeVertexCount = 6u;
        const double polygon[6][2]{{0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 4}, {0, 4}};
        for (unsigned i = 0; i < 6; ++i)
            for (unsigned j = 0; j < 2; ++j) volume.customVolumeVertices[i][j] = polygon[i][j];
        auto check = [&](double ax, double ay, double az,
                         double bx, double by, double bz, double expected) {
            double fraction = -1.0;
            const bool hit = cuda::custom_volume::SegmentEntry(
                    volume, ax, ay, az, bx, by, bz, &fraction);
            if (hit != (expected >= 0.0) ||
                (hit && std::abs(fraction - expected) > 1e-12))
                throw std::runtime_error("custom volume intersection mismatch");
        };
        check(-1, 2, 1, 5, 2, 1, 1.0 / 6.0); // Outside-to-outside, concave arm.
        check(2, 2, 1, 3, 3, 1, -1);          // Concave notch stays outside.
        check(0.5, 0.5, -1, 0.5, 0.5, 3, 0.25); // Extrusion cap.
        check(-1, 0, 1, 5, 0, 1, 1.0 / 6.0); // Collinear edge.
        check(-1, 1, 1, 1, -1, 1, 0.5);      // Single-point tangency.
        check(-1, 2, 3, 5, 2, 3, -1);        // Above extrusion.
        for (unsigned plane = 0; plane < 3; ++plane) {
            volume.optionFlags = plane;
            const double origin[3]{100, 200, 300};
            for (unsigned i = 0; i < 3; ++i) volume.values[i] = origin[i];
            if (plane == 0) check(99, 202, 301, 105, 202, 301, 1.0 / 6.0);
            if (plane == 1) check(99, 201, 302, 105, 201, 302, 1.0 / 6.0);
            if (plane == 2) check(101, 199, 302, 101, 205, 302, 1.0 / 6.0);
        }
        volume.optionFlags = 0;
        volume.values[0] = volume.values[1] = volume.values[2] = 0;
        volume.customVolumeVertexCount = 256;
        for (unsigned i = 0; i < 256; ++i) {
            const double angle = 6.283185307179586 * i / 256;
            volume.customVolumeVertices[i][0] = std::cos(angle);
            volume.customVolumeVertices[i][1] = std::sin(angle);
        }
        check(-2, 0, 1, 2, 0, 1, 0.25);
        std::cout << "custom volume: concavity, caps, tangency, planes, origin, 256 vertices passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
