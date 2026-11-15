#ifndef FOREVERVALIDATOR_EXPERIMENTAL_AROUND_WALLS_TABLE_H
#define FOREVERVALIDATOR_EXPERIMENTAL_AROUND_WALLS_TABLE_H

// The Distance target's way around walls, for the GPU: a graph built on the
// CPU (ForeverTAS's AroundWallsDistance) packed into one table per trigger,
// and its query, the same code on the host and on the device. The table is
// plain bytes: a Header, then arrays at the header's byte offsets.
//
// Corners are filled on demand: a corner not filled yet (kCornerUnfilled)
// that a query needs is marked in the table's missed bits and the query's
// answer is not to be used; whoever ran it fills the missed corners
// (FillCorner, as AroundWallsDistance fills its own) and asks again.

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define FV_AROUND_WALLS_HD __host__ __device__
#else
#include <cmath>
#define FV_AROUND_WALLS_HD
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace forevervalidator::experimental::around_walls {

inline constexpr std::uint32_t kMagic = 0x41574454u;  // "TDWA"
inline constexpr std::uint32_t kVersion = 2u;
inline constexpr std::uint32_t kNone = 0xffffffffu;
inline constexpr std::uint32_t kCornerUnfilled = 2u;  // Corner::direct

// The query's constants (AroundWallsDistance's).
inline constexpr int kSightResolution = 4;
inline constexpr int kSightTexels = 6 * kSightResolution * kSightResolution;
inline constexpr double kSightUnit = 0.01;
inline constexpr std::uint16_t kSightOpen = 0xffffu;
inline constexpr double kGridCell = 2.0;
inline constexpr int kOuterResolution = 24;
inline constexpr int kBandCount = 13;
inline constexpr int kCornerNodes = 3;
inline constexpr int kFloorPoints = 8;
inline constexpr int kFallbackSweeps = 48;
inline constexpr int kCornerSweeps = 512;
inline constexpr double kSampleClearance = 0.75;
inline constexpr double kCrossingTolerance = 0.01;
inline constexpr double kNearerOnTheWay = 1e-6;
FV_AROUND_WALLS_HD inline double Band(int band) {
    return band == 0 ? 0.0 : static_cast<double>(1u << band);
}

struct Header {
    std::uint32_t magic = kMagic;
    std::uint32_t version = kVersion;
    double radius = 0.0;
    double boxMinimum[3]{};
    double boxMaximum[3]{};
    double center[3]{};
    std::int32_t cells[3]{};
    std::uint32_t triggerCount = 0u;
    std::uint32_t wallCount = 0u;
    std::uint32_t treeCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t floorCellCount = 0u;
    std::uint64_t outerStart = 0u;
    std::uint64_t floorStart = 0u;
    std::uint64_t cornerCount = 0u;
    // Byte offsets from the table's start.
    std::uint64_t triggers = 0u;     // double[9] per triangle
    std::uint64_t walls = 0u;        // double[9] per triangle
    std::uint64_t planes = 0u;       // Plane per wall
    std::uint64_t tree = 0u;         // TreeNode, depth first
    std::uint64_t nodes = 0u;        // double[3] per node
    std::uint64_t lengths = 0u;      // double per node
    std::uint64_t sight = 0u;        // uint16[kSightTexels] per node
    std::uint64_t corners = 0u;      // Corner per corner
    std::uint64_t floors = 0u;       // double[3] per floor point
    std::uint64_t floorCells = 0u;   // FloorCell, sorted by x, y, z
    std::uint64_t floorIndices = 0u; // uint32 per floor point in a cell
    std::uint64_t missed = 0u;       // uint32: any, then one bit per corner
    std::uint64_t size = 0u;
};

// A wall's plane (hasNormal 0 for a sliver), box and bounding sphere.
struct Plane {
    double normal[3];
    double hasNormal;
    double offset;
    double boxMinimum[3];
    double boxMaximum[3];
    double center[3];
    double bound;
};

struct TreeNode {
    double boxMinimum[3];
    double boxMaximum[3];
    std::uint32_t first;
    std::uint32_t count;  // leaf when non-zero
    std::uint32_t right;
    std::uint32_t unused;
};

struct Corner {
    std::uint32_t direct;  // 0, 1, or kCornerUnfilled
    std::uint32_t nodes[kCornerNodes];
};

FV_AROUND_WALLS_HD inline std::uint64_t MissedBytes(std::uint64_t cornerCount) {
    return 4u * (1u + (cornerCount + 31u) / 32u);
}

struct FloorCell {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;
    std::uint32_t first;
    std::uint32_t count;
};

struct Path {
    double length = 0.0;
    double via[3]{};
    bool inSight = false;
};

namespace detail {

struct V {
    double x, y, z;
};
FV_AROUND_WALLS_HD inline V Add(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
FV_AROUND_WALLS_HD inline V Sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
FV_AROUND_WALLS_HD inline V Mul(V a, double s) { return {a.x * s, a.y * s, a.z * s}; }
FV_AROUND_WALLS_HD inline double Dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
FV_AROUND_WALLS_HD inline V Cross(V a, V b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
FV_AROUND_WALLS_HD inline double Length(V a) { return sqrt(Dot(a, a)); }
FV_AROUND_WALLS_HD inline double Min(double a, double b) { return a < b ? a : b; }
FV_AROUND_WALLS_HD inline double Max(double a, double b) { return a > b ? a : b; }
FV_AROUND_WALLS_HD inline double Clamp(double v, double lo, double hi) {
    return v < lo ? lo : v > hi ? hi : v;
}
FV_AROUND_WALLS_HD inline int ClampInt(int v, int lo, int hi) {
    return v < lo ? lo : v > hi ? hi : v;
}
FV_AROUND_WALLS_HD inline double Axis(V v, int axis) {
    return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
}
FV_AROUND_WALLS_HD inline V At(const double *p) { return {p[0], p[1], p[2]}; }

struct Tri {
    V a, b, c;
};
FV_AROUND_WALLS_HD inline Tri TriAt(const double *p) {
    return {{p[0], p[1], p[2]}, {p[3], p[4], p[5]}, {p[6], p[7], p[8]}};
}

FV_AROUND_WALLS_HD inline V ClosestOnTriangle(V p, const Tri &t) {
    const V ab = Sub(t.b, t.a);
    const V ac = Sub(t.c, t.a);
    const V ap = Sub(p, t.a);
    const double d1 = Dot(ab, ap);
    const double d2 = Dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return t.a;
    const V bp = Sub(p, t.b);
    const double d3 = Dot(ab, bp);
    const double d4 = Dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return t.b;
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return Add(t.a, Mul(ab, d1 / (d1 - d3)));
    const V cp = Sub(p, t.c);
    const double d5 = Dot(ab, cp);
    const double d6 = Dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return t.c;
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return Add(t.a, Mul(ac, d2 / (d2 - d6)));
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return Add(t.b, Mul(Sub(t.c, t.b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    const double denominator = 1.0 / (va + vb + vc);
    return Add(t.a, Add(Mul(ab, vb * denominator), Mul(ac, vc * denominator)));
}

FV_AROUND_WALLS_HD inline double PointTriangleDistance(V p, const Tri &t) {
    return Length(Sub(p, ClosestOnTriangle(p, t)));
}

FV_AROUND_WALLS_HD inline double SegmentSegmentDistance(V p1, V q1, V p2, V q2) {
    const V d1 = Sub(q1, p1);
    const V d2 = Sub(q2, p2);
    const V r = Sub(p1, p2);
    const double a = Dot(d1, d1);
    const double e = Dot(d2, d2);
    const double f = Dot(d2, r);
    double s = 0.0;
    double t = 0.0;
    if (a <= 1e-12 && e <= 1e-12) return Length(r);
    if (a <= 1e-12) {
        t = Clamp(f / e, 0.0, 1.0);
    } else {
        const double c = Dot(d1, r);
        if (e <= 1e-12) {
            s = Clamp(-c / a, 0.0, 1.0);
        } else {
            const double b = Dot(d1, d2);
            const double denominator = a * e - b * b;
            s = denominator > 1e-12 ? Clamp((b * f - c * e) / denominator, 0.0, 1.0) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = Clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = Clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    return Length(Sub(Add(p1, Mul(d1, s)), Add(p2, Mul(d2, t))));
}

FV_AROUND_WALLS_HD inline bool Crosses(V p, V q, const Tri &t) {
    const V direction = Sub(q, p);
    const double length = Length(direction);
    if (length <= 2.0 * kCrossingTolerance) return false;
    const V e1 = Sub(t.b, t.a);
    const V e2 = Sub(t.c, t.a);
    const V h = Cross(direction, e2);
    const double a = Dot(e1, h);
    if (fabs(a) < 1e-12) return false;
    const double f = 1.0 / a;
    const V s = Sub(p, t.a);
    const double u = f * Dot(s, h);
    if (u < 0.0 || u > 1.0) return false;
    const V qv = Cross(s, e1);
    const double v = f * Dot(direction, qv);
    if (v < 0.0 || u + v > 1.0) return false;
    const double along = f * Dot(e2, qv);
    const double margin = kCrossingTolerance / length;
    return along > margin && along < 1.0 - margin;
}

FV_AROUND_WALLS_HD inline double PointSegmentDistance(V point, V p, V q) {
    const V d = Sub(q, p);
    const double length = Dot(d, d);
    const double t = length > 1e-12 ? Clamp(Dot(Sub(point, p), d) / length, 0.0, 1.0) : 0.0;
    return Length(Sub(point, Add(p, Mul(d, t))));
}

FV_AROUND_WALLS_HD inline bool BoxesOverlap(const double *aMin, const double *aMax,
                                            V bMin, V bMax) {
    return aMin[0] <= bMax.x && bMin.x <= aMax[0] && aMin[1] <= bMax.y &&
           bMin.y <= aMax[1] && aMin[2] <= bMax.z && bMin.z <= aMax[2];
}

FV_AROUND_WALLS_HD inline bool Stops(V p, V q, double radius, V reachMin, V reachMax,
                                     const Tri &t, const Plane &plane) {
    if (!BoxesOverlap(plane.boxMinimum, plane.boxMaximum, reachMin, reachMax)) return false;
    if (plane.hasNormal != 0.0) {
        const V normal = At(plane.normal);
        const double fromP = Dot(normal, p) - plane.offset;
        const double fromQ = Dot(normal, q) - plane.offset;
        if ((fromP >= radius && fromQ >= radius) || (fromP <= -radius && fromQ <= -radius)) {
            return false;
        }
        if ((fromP > 0.0) != (fromQ > 0.0) && Crosses(p, q, t)) return true;
    } else if (Crosses(p, q, t)) {
        return true;
    }
    if (PointSegmentDistance(At(plane.center), p, q) >= plane.bound + radius) return false;
    const double edges = Min(Min(SegmentSegmentDistance(p, q, t.a, t.b),
                                 SegmentSegmentDistance(p, q, t.b, t.c)),
                             SegmentSegmentDistance(p, q, t.c, t.a));
    if (edges >= radius) return false;
    return edges < Min(PointTriangleDistance(p, t), PointTriangleDistance(q, t)) - kNearerOnTheWay;
}

FV_AROUND_WALLS_HD inline bool SegmentThroughBox(const TreeNode &node, V p, V step, V inverse,
                                                 double grow) {
    double enter = 0.0;
    double leave = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double start = Axis(p, axis);
        const double low = node.boxMinimum[axis] - grow;
        const double high = node.boxMaximum[axis] + grow;
        if (fabs(Axis(step, axis)) < 1e-12) {
            if (start < low || start > high) return false;
            continue;
        }
        const double scale = Axis(inverse, axis);
        double t0 = (low - start) * scale;
        double t1 = (high - start) * scale;
        if (t0 > t1) {
            const double swap = t0;
            t0 = t1;
            t1 = swap;
        }
        enter = Max(enter, t0);
        leave = Min(leave, t1);
        if (enter > leave) return false;
    }
    return true;
}

struct Table {
    const unsigned char *bytes;
    const Header *header;
    FV_AROUND_WALLS_HD explicit Table(const unsigned char *data)
        : bytes(data), header(reinterpret_cast<const Header *>(data)) {}
    template <typename T>
    FV_AROUND_WALLS_HD const T *Array(std::uint64_t offset) const {
        return reinterpret_cast<const T *>(bytes + offset);
    }
    FV_AROUND_WALLS_HD Tri Wall(std::uint32_t i) const {
        return TriAt(Array<double>(header->walls) + 9u * i);
    }
    FV_AROUND_WALLS_HD V Node(std::uint32_t i) const {
        return At(Array<double>(header->nodes) + 3u * i);
    }
    FV_AROUND_WALLS_HD double Key(V p, std::uint32_t node) const {
        return Length(Sub(p, Node(node))) + Array<double>(header->lengths)[node];
    }

    // A wall that stops a sphere moving from p to q, or kNone.
    FV_AROUND_WALLS_HD std::uint32_t Blocker(V p, V q) const {
        if (header->treeCount == 0u) return kNone;
        const double radius = header->radius;
        const V reachMin{Min(p.x, q.x) - radius, Min(p.y, q.y) - radius, Min(p.z, q.z) - radius};
        const V reachMax{Max(p.x, q.x) + radius, Max(p.y, q.y) + radius, Max(p.z, q.z) + radius};
        const V step = Sub(q, p);
        const V inverse{fabs(step.x) < 1e-12 ? 0.0 : 1.0 / step.x,
                        fabs(step.y) < 1e-12 ? 0.0 : 1.0 / step.y,
                        fabs(step.z) < 1e-12 ? 0.0 : 1.0 / step.z};
        const TreeNode *tree = Array<TreeNode>(header->tree);
        const Plane *planes = Array<Plane>(header->planes);
        std::uint32_t stack[64];
        int depth = 0;
        stack[depth++] = 0u;
        while (depth > 0) {
            const std::uint32_t index = stack[--depth];
            const TreeNode &node = tree[index];
            if (!SegmentThroughBox(node, p, step, inverse, radius)) continue;
            if (node.count != 0u) {
                for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
                    if (Stops(p, q, radius, reachMin, reachMax, Wall(i), planes[i])) return i;
                }
                continue;
            }
            stack[depth++] = node.right;
            stack[depth++] = index + 1u;
        }
        return kNone;
    }

    FV_AROUND_WALLS_HD bool Hidden(std::uint32_t node, V to) const {
        const V from = Node(node);
        const V d = Sub(to, from);
        const double ax = fabs(d.x);
        const double ay = fabs(d.y);
        const double az = fabs(d.z);
        int face;
        double u;
        double v;
        if (ax >= ay && ax >= az) {
            face = d.x >= 0.0 ? 0 : 1;
            u = d.y / ax;
            v = d.z / ax;
        } else if (ay >= az) {
            face = d.y >= 0.0 ? 2 : 3;
            u = d.x / ay;
            v = d.z / ay;
        } else {
            face = d.z >= 0.0 ? 4 : 5;
            u = d.x / az;
            v = d.y / az;
        }
        const int i = ClampInt(static_cast<int>((u + 1.0) * 0.5 * kSightResolution), 0,
                               kSightResolution - 1);
        const int j = ClampInt(static_cast<int>((v + 1.0) * 0.5 * kSightResolution), 0,
                               kSightResolution - 1);
        const std::uint16_t depth = Array<std::uint16_t>(header->sight)
                [static_cast<std::uint64_t>(node) * kSightTexels +
                 (static_cast<std::uint64_t>(face) * kSightResolution + i) * kSightResolution + j];
        return depth != kSightOpen && Length(d) > depth * kSightUnit;
    }

    FV_AROUND_WALLS_HD V ClosestOnTrigger(V p) const {
        V best{0.0, 0.0, 0.0};
        double bestDistance = 1e300;
        const double *triggers = Array<double>(header->triggers);
        for (std::uint32_t i = 0; i < header->triggerCount; ++i) {
            const V closest = ClosestOnTriangle(p, TriAt(triggers + 9u * i));
            const double distance = Length(Sub(p, closest));
            if (distance < bestDistance) {
                bestDistance = distance;
                best = closest;
            }
        }
        return best;
    }

    FV_AROUND_WALLS_HD bool InBox(V p) const {
        const double *lo = header->boxMinimum;
        const double *hi = header->boxMaximum;
        return p.x >= lo[0] && p.x <= hi[0] && p.y >= lo[1] && p.y <= hi[1] && p.z >= lo[2] &&
               p.z <= hi[2];
    }

    FV_AROUND_WALLS_HD V Exit(V direction) const {
        double nearest = 1e300;
        for (int axis = 0; axis < 3; ++axis) {
            const double step = Axis(direction, axis);
            if (fabs(step) < 1e-12) continue;
            const double wall = step > 0.0 ? header->boxMaximum[axis] : header->boxMinimum[axis];
            nearest = Min(nearest, (wall - header->center[axis]) / step);
        }
        return Add(At(header->center), Mul(direction, nearest));
    }

    FV_AROUND_WALLS_HD void CornersAround(V p, std::uint64_t (&corners)[8]) const {
        const std::int32_t *cells = header->cells;
        if (InBox(p)) {
            int base[3];
            for (int axis = 0; axis < 3; ++axis) {
                base[axis] = ClampInt(static_cast<int>((Axis(p, axis) - header->boxMinimum[axis]) /
                                                       kGridCell),
                                      0, cells[axis] - 1);
            }
            for (int c = 0; c < 8; ++c) {
                const std::uint64_t i = static_cast<std::uint64_t>(base[0] + (c & 1));
                const std::uint64_t j = static_cast<std::uint64_t>(base[1] + (c >> 1 & 1));
                const std::uint64_t k = static_cast<std::uint64_t>(base[2] + (c >> 2 & 1));
                corners[c] = (i * static_cast<std::uint64_t>(cells[1] + 1) + j) *
                                     static_cast<std::uint64_t>(cells[2] + 1) + k;
            }
            return;
        }
        V direction = Sub(p, At(header->center));
        const double length = Length(direction);
        direction = length > 0.0 ? Mul(direction, 1.0 / length) : V{1.0, 0.0, 0.0};
        const double beyond = Length(Sub(p, Exit(direction)));
        int band = 0;
        while (band + 2 < kBandCount && Band(band + 1) <= beyond) ++band;
        const double ax = fabs(direction.x);
        const double ay = fabs(direction.y);
        const double az = fabs(direction.z);
        int face;
        double u;
        double v;
        if (ax >= ay && ax >= az) {
            face = direction.x >= 0.0 ? 0 : 1;
            u = direction.y / ax;
            v = direction.z / ax;
        } else if (ay >= az) {
            face = direction.y >= 0.0 ? 2 : 3;
            u = direction.x / ay;
            v = direction.z / ay;
        } else {
            face = direction.z >= 0.0 ? 4 : 5;
            u = direction.x / az;
            v = direction.y / az;
        }
        const int i = ClampInt(static_cast<int>((u + 1.0) * 0.5 * kOuterResolution), 0,
                               kOuterResolution - 1);
        const int j = ClampInt(static_cast<int>((v + 1.0) * 0.5 * kOuterResolution), 0,
                               kOuterResolution - 1);
        for (int c = 0; c < 8; ++c) {
            corners[c] = header->outerStart +
                         ((static_cast<std::uint64_t>(face) * (kOuterResolution + 1) +
                           static_cast<std::uint64_t>(i + (c & 1))) *
                                  (kOuterResolution + 1) +
                          static_cast<std::uint64_t>(j + (c >> 1 & 1))) *
                                 kBandCount +
                         static_cast<std::uint64_t>(band + (c >> 2 & 1));
        }
    }

    FV_AROUND_WALLS_HD int FloorPointsAround(V p, std::uint64_t (&points)[kFloorPoints]) const {
        int count = 0;
        const std::int64_t bx = static_cast<std::int64_t>(floor((p.x - kGridCell * 0.5) / kGridCell));
        const std::int64_t by = static_cast<std::int64_t>(floor((p.y - kGridCell * 0.5) / kGridCell));
        const std::int64_t bz = static_cast<std::int64_t>(floor((p.z - kGridCell * 0.5) / kGridCell));
        const FloorCell *cells = Array<FloorCell>(header->floorCells);
        const std::uint32_t *indices = Array<std::uint32_t>(header->floorIndices);
        for (int c = 0; c < 8; ++c) {
            const std::int64_t x = bx + (c & 1);
            const std::int64_t y = by + (c >> 1 & 1);
            const std::int64_t z = bz + (c >> 2 & 1);
            std::uint32_t lo = 0u;
            std::uint32_t hi = header->floorCellCount;
            while (lo < hi) {
                const std::uint32_t mid = (lo + hi) / 2u;
                const FloorCell &cell = cells[mid];
                const bool less = cell.x != x ? cell.x < x : cell.y != y ? cell.y < y : cell.z < z;
                if (less) lo = mid + 1u; else hi = mid;
            }
            if (lo == header->floorCellCount || cells[lo].x != x || cells[lo].y != y ||
                cells[lo].z != z) {
                continue;
            }
            for (std::uint32_t k = 0; k < cells[lo].count; ++k) {
                if (count == kFloorPoints) return count;
                points[count++] = header->floorStart + indices[cells[lo].first + k];
            }
        }
        return count;
    }

    // Whether any wall comes nearer p than distance.
    FV_AROUND_WALLS_HD bool AnyWithin(V p, double distance) const {
        if (header->treeCount == 0u) return false;
        const TreeNode *tree = Array<TreeNode>(header->tree);
        std::uint32_t stack[64];
        int depth = 0;
        stack[depth++] = 0u;
        while (depth > 0) {
            const std::uint32_t index = stack[--depth];
            const TreeNode &node = tree[index];
            const double dx = Max(Max(node.boxMinimum[0] - p.x, 0.0), p.x - node.boxMaximum[0]);
            const double dy = Max(Max(node.boxMinimum[1] - p.y, 0.0), p.y - node.boxMaximum[1]);
            const double dz = Max(Max(node.boxMinimum[2] - p.z, 0.0), p.z - node.boxMaximum[2]);
            if (sqrt(dx * dx + dy * dy + dz * dz) >= distance) continue;
            if (node.count != 0u) {
                for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
                    if (PointTriangleDistance(p, Wall(i)) < distance) return true;
                }
                continue;
            }
            stack[depth++] = node.right;
            stack[depth++] = index + 1u;
        }
        return false;
    }

    // Where corner sits: the inner grid's points, then the outer table's
    // (face, u, v, band), then the floor points.
    FV_AROUND_WALLS_HD V CornerPoint(std::uint64_t corner) const {
        if (corner >= header->floorStart) {
            return At(Array<double>(header->floors) + 3u * (corner - header->floorStart));
        }
        const std::int32_t *cells = header->cells;
        if (corner < header->outerStart) {
            const std::uint64_t layer = static_cast<std::uint64_t>(cells[1] + 1) * (cells[2] + 1);
            const int i = static_cast<int>(corner / layer);
            const int j = static_cast<int>(corner % layer / (cells[2] + 1));
            const int k = static_cast<int>(corner % (cells[2] + 1));
            return Add(At(header->boxMinimum), V{i * kGridCell, j * kGridCell, k * kGridCell});
        }
        std::uint64_t rest = corner - header->outerStart;
        const int band = static_cast<int>(rest % kBandCount);
        rest /= kBandCount;
        const int v = static_cast<int>(rest % (kOuterResolution + 1));
        rest /= kOuterResolution + 1;
        const int u = static_cast<int>(rest % (kOuterResolution + 1));
        const int face = static_cast<int>(rest / (kOuterResolution + 1));
        const double sign = face % 2 == 0 ? 1.0 : -1.0;
        const double a = 2.0 * u / kOuterResolution - 1.0;
        const double b = 2.0 * v / kOuterResolution - 1.0;
        V direction = face / 2 == 0 ? V{sign, a, b} : face / 2 == 1 ? V{a, sign, b} : V{a, b, sign};
        direction = Mul(direction, 1.0 / Length(direction));
        return Add(Exit(direction), Mul(direction, Band(band)));
    }

    // Every node, cheapest first, until `want` are in sight of p or after
    // `sweeps` sweeps: whether the trigger's closest point is in plain
    // sight, and the nodes found (their count). Nodes come in (key, index)
    // order, a batch of the next cheapest per pass over them.
    FV_AROUND_WALLS_HD int SearchMany(V p, bool &direct, std::uint32_t *result, int want,
                                      int sweeps) const {
        direct = Blocker(p, ClosestOnTrigger(p)) == kNone;
        int wanted = 0;
        constexpr int kBatch = 32;
        double lastKey = -1e300;
        std::uint32_t lastNode = kNone;
        while (sweeps > 0) {
            double keys[kBatch];
            std::uint32_t found[kBatch];
            int count = 0;
            for (std::uint32_t i = 0; i < header->nodeCount; ++i) {
                const double key = Key(p, i);
                const bool after = key > lastKey ||
                        (key == lastKey && (lastNode == kNone || i > lastNode));
                if (!after) continue;
                if (count == kBatch &&
                    (key > keys[kBatch - 1] || (key == keys[kBatch - 1] && i > found[kBatch - 1]))) {
                    continue;
                }
                int at = count < kBatch ? count++ : kBatch - 1;
                while (at > 0 && (keys[at - 1] > key || (keys[at - 1] == key && found[at - 1] > i))) {
                    keys[at] = keys[at - 1];
                    found[at] = found[at - 1];
                    --at;
                }
                keys[at] = key;
                found[at] = i;
            }
            if (count == 0) return wanted;
            for (int j = 0; j < count && sweeps > 0; ++j) {
                lastKey = keys[j];
                lastNode = found[j];
                if (Hidden(found[j], p)) continue;
                --sweeps;
                if (Blocker(Node(found[j]), p) == kNone) {
                    result[wanted++] = found[j];
                    if (wanted == want) return wanted;
                }
            }
        }
        return wanted;
    }
    FV_AROUND_WALLS_HD std::uint32_t Search(V p, bool &direct, int sweeps) const {
        std::uint32_t found = kNone;
        return SearchMany(p, direct, &found, 1, sweeps) == 1 ? found : kNone;
    }
};

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
// Host: the corners this thread's queries marked since it last cleared the
// count (several threads may query one table: each learns of its own).
inline std::uint32_t &HostMisses() {
    static thread_local std::uint32_t misses = 0u;
    return misses;
}
#endif

// A corner a query needed before it was filled. The table's bytes are the
// query's to mark even where it reads them as const.
FV_AROUND_WALLS_HD inline void MarkMissed(const unsigned char *bytes, std::uint64_t corner) {
    const Header *header = reinterpret_cast<const Header *>(bytes);
    auto *words = reinterpret_cast<std::uint32_t *>(const_cast<unsigned char *>(bytes) +
                                                    header->missed);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    atomicOr(words + 1u + corner / 32u, 1u << (corner % 32u));
    atomicOr(words, 1u);
#elif defined(_MSC_VER) && !defined(__clang__)
    _InterlockedOr(reinterpret_cast<volatile long *>(words + 1u + corner / 32u),
                   static_cast<long>(1u << (corner % 32u)));
    _InterlockedOr(reinterpret_cast<volatile long *>(words), 1L);
    ++HostMisses();
#else
    __atomic_fetch_or(words + 1u + corner / 32u, 1u << (corner % 32u), __ATOMIC_RELAXED);
    __atomic_fetch_or(words, 1u, __ATOMIC_RELAXED);
    ++HostMisses();
#endif
}

}  // namespace detail

// Fills one corner (of the header's cornerCount): its cheapest nodes in
// sight and whether the trigger is in plain sight from it; nothing from
// inside walls. AroundWallsDistance's own corners, byte for byte.
FV_AROUND_WALLS_HD inline void FillCorner(unsigned char *bytes, std::uint64_t corner) {
    const detail::Table table(bytes);
    Corner found{};
    for (int n = 0; n < kCornerNodes; ++n) found.nodes[n] = kNone;
    const detail::V point = table.CornerPoint(corner);
    if (!table.AnyWithin(point, table.header->radius * kSampleClearance)) {
        bool direct = false;
        table.SearchMany(point, direct, found.nodes, kCornerNodes, kCornerSweeps);
        found.direct = direct ? 1u : 0u;
    }
    reinterpret_cast<Corner *>(bytes + table.header->corners)[corner] = found;
}

// Whether bytes hold a table this code reads.
inline bool Valid(const unsigned char *bytes, std::uint64_t size) {
    if (size < sizeof(Header)) return false;
    const Header *header = reinterpret_cast<const Header *>(bytes);
    return header->magic == kMagic && header->version == kVersion && header->size == size;
}

// Path from a sphere of the table's radius at (x, y, z) until it touches
// the trigger: AroundWallsDistance::PathTo without the memory of earlier
// queries.
FV_AROUND_WALLS_HD inline Path PathTo(const unsigned char *bytes, double x, double y, double z) {
    using namespace detail;
    const Table table(bytes);
    const Header &header = *table.header;
    const V center{x, y, z};
    const Corner *corners = table.Array<Corner>(header.corners);
    std::uint64_t around[8];
    table.CornersAround(center, around);
    std::uint64_t floorPoints[kFloorPoints];
    const int floorCount = table.FloorPointsAround(center, floorPoints);
    double keys[(8 + kFloorPoints) * kCornerNodes];
    std::uint32_t nodes[(8 + kFloorPoints) * kCornerNodes];
    int count = 0;
    bool direct = false;
    const auto add = [&](std::uint32_t node) {
        if (node == kNone) return;
        for (int i = 0; i < count; ++i) {
            if (nodes[i] == node) return;
        }
        // Kept sorted by (key, node).
        const double key = table.Key(center, node);
        int at = count++;
        while (at > 0 && (keys[at - 1] > key || (keys[at - 1] == key && nodes[at - 1] > node))) {
            keys[at] = keys[at - 1];
            nodes[at] = nodes[at - 1];
            --at;
        }
        keys[at] = key;
        nodes[at] = node;
    };
    for (int c = 0; c < 8; ++c) {
        const Corner &corner = corners[around[c]];
        if (corner.direct == kCornerUnfilled) MarkMissed(bytes, around[c]);
        direct |= corner.direct == 1u;
        if (floorCount == 0) {
            for (int n = 0; n < kCornerNodes; ++n) add(corner.nodes[n]);
        }
    }
    for (int f = 0; f < floorCount; ++f) {
        const Corner &corner = corners[floorPoints[f]];
        if (corner.direct == kCornerUnfilled) MarkMissed(bytes, floorPoints[f]);
        direct |= corner.direct == 1u;
        for (int n = 0; n < kCornerNodes; ++n) add(corner.nodes[n]);
    }
    Path path;
    if (direct) {
        const V closest = table.ClosestOnTrigger(center);
        if (table.Blocker(center, closest) == kNone) {
            path.length = Max(0.0, Length(Sub(center, closest)) - header.radius);
            path.via[0] = closest.x;
            path.via[1] = closest.y;
            path.via[2] = closest.z;
            path.inSight = true;
            return path;
        }
    }
    for (int i = 0; i < count; ++i) {
        const std::uint32_t node = nodes[i];
        if (table.Hidden(node, center)) continue;
        if (table.Blocker(table.Node(node), center) != kNone) continue;
        const V at = table.Node(node);
        path.length = Max(0.0, keys[i]);
        path.via[0] = at.x;
        path.via[1] = at.y;
        path.via[2] = at.z;
        return path;
    }
    bool inSight = false;
    const std::uint32_t found = table.Search(center, inSight, kFallbackSweeps);
    if (found != kNone && !inSight) {
        const V at = table.Node(found);
        path.length = Max(0.0, table.Key(center, found));
        path.via[0] = at.x;
        path.via[1] = at.y;
        path.via[2] = at.z;
        return path;
    }
    const V closest = table.ClosestOnTrigger(center);
    path.length = Max(0.0, Length(Sub(center, closest)) - header.radius);
    path.via[0] = closest.x;
    path.via[1] = closest.y;
    path.via[2] = closest.z;
    path.inSight = inSight;
    if (inSight || count == 0) return path;
    // Nothing found in sight: the cheapest way the points around saw.
    const V at = table.Node(nodes[0]);
    path.length = Max(0.0, keys[0]);
    path.via[0] = at.x;
    path.via[1] = at.y;
    path.via[2] = at.z;
    path.inSight = false;
    return path;
}

}  // namespace forevervalidator::experimental::around_walls

#endif
