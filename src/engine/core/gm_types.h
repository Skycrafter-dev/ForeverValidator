#pragma once

#include "engine/core/engine_types.h"
struct GmMat3;
struct GmIso4;
struct SMeshOctreeCell;
using GmMeshOctreeCell = SMeshOctreeCell;

enum class GmAxis : u32 {
    X = 0u,
    Y = 1u,
    Z = 2u,
};

struct GmVec3 {
    float x;
    float y;
    float z;
    void SetMult(const GmVec3 &source, const GmMat3 &mat);
    void SetMult(const GmVec3 &source, const GmIso4 &iso);
    void SetMultTranspose(const GmVec3 &source, const GmMat3 &mat);
    void Mult(const GmIso4 &iso);
    void Mult(const GmMat3 &mat);
    void MultTranspose(const GmMat3 &mat);
    void MultInverse(const GmIso4 &iso);
    unsigned long IsNearlyEqual(const GmVec3 &rhs) const;
    void NormalizeIfAboveAngleEpsilon(void);
    float Dot(const GmVec3 &rhs) const;
    float Component(GmAxis axis) const;
    float LengthSquaredForCollision(void) const;
    float LengthSquaredYXZ(void) const;
    GmVec3 Negated(void) const;
    GmVec3 operator-(const GmVec3 &rhs) const;
    GmVec3 Cross(const GmVec3 &rhs) const;
    GmVec3 SubtractForCollision(const GmVec3 &rhs) const;
    GmVec3 AddForCollision(const GmVec3 &rhs) const;
    GmVec3 ScaleForCollision(float scale) const;
    float MinComponentForGmSurf(void) const;
    float MaxComponentForGmSurf(void) const;
    GmVec3 NormalizeForCollision(float epsilonSq) const;
    void MultEllipsoidMeshWorldNormalForGmSurf(const GmMat3 &rotation);
    void TransformAndNormalizeDirectionForGmSurf(const GmIso4 &iso, float epsilonSq);
    void MultInverseIso4LocalForGmSurf(const GmIso4 &iso);
    GmVec3 LocalToWorldMat3ForSolveImpulse(const GmMat3 &rotation) const;
    GmVec3 LocalToWorldMat3SideAForSolveImpulse(
            const GmMat3 &rotation) const;
    static GmVec3 ZeroForComputeCorpusForces(void);
    void ScaleInPlaceForComputeCorpusForces(float scale);
    void AccumulateForComputeCorpusForces(const GmVec3 &value);
    void AddScaledForComputeCorpusForces(GmVec3 value, float scale);
    float PhysicsStep2LinearSpeedLength(void) const;
    float PhysicsStep2AngularSpeedLength(void) const;
    static GmVec3 Zero(void);
    static float GetAngle(const GmVec3 &lhs, const GmVec3 &rhs);
};

struct GmVec2 {
    float x;
    float y;
};

struct GmVec4 {
    float x;
    float y;
    float z;
    float w;
    unsigned long PlaneEqIsNearlyEqual(
            const GmVec4 &other,
            float normalDotThreshold,
            float distanceThreshold) const;
    void PlaneEqMult(const GmIso4 &iso);
};

enum class ECardinalDir : u32 {
    North = 0u,
    West = 1u,
    South = 2u,
    East = 3u,
};

enum class EBlockType : u32 {
    Flat = 0u,
    Frontier = 1u,
    Classic = 2u,
    Road = 3u,
    Clip = 4u,
    Pylon = 5u,
    Slope = 6u,
    RectAsym = 7u,
};

enum class ETerrain : u32 {
    Underground = 0u,
    Ground = 1u,
    Reserved = 2u,
    Air = 3u,
    Outside = 4u,
};

struct GmNat3 {
    u32 x;
    u32 y;
    u32 z;
};

struct GmInt3 {
    int32_t x;
    int32_t y;
    int32_t z;
};

struct GmNat2 {
    u32 x;
    u32 y;
};

struct GmQuat {
    float w;
    float x;
    float y;
    float z;
    void Set(const GmQuat &rhs);
    void SetIdentity(void);
    void SetMult(const GmQuat &lhs, const GmQuat &rhs);
    void Mult(const GmQuat &rhs);
    void SetInverse(const GmQuat &rhs);
    void Normalize(void);
    void Set(const GmMat3 &mat);
    void SetSlerp(GmQuat from, const GmQuat &to, float blend);
    void GetRotation(float &angle, GmVec3 &axis);
};

template <typename Value>
struct GmSpring;

template <>
struct GmSpring<float> {
    float stiffness;
    float damping;
    float value;
    float target;
    float velocity;

    GmSpring(void);
    float GetCriticalKa(void);
    void ClearVals(void);
    void Integrate(float dt);
};

struct GmMat3 {
    GmVec3 basisX{};
    GmVec3 basisY{};
    GmVec3 basisZ{};

    GmVec3 &Basis(GmAxis axis) {
        if (axis == GmAxis::X) {
            return basisX;
        }
        return axis == GmAxis::Y ? basisY : basisZ;
    }
    const GmVec3 &Basis(GmAxis axis) const {
        if (axis == GmAxis::X) {
            return basisX;
        }
        return axis == GmAxis::Y ? basisY : basisZ;
    }
    GmVec3 Row(GmAxis axis) const {
        if (axis == GmAxis::X) {
            return {basisX.x, basisY.x, basisZ.x};
        }
        if (axis == GmAxis::Y) {
            return {basisX.y, basisY.y, basisZ.y};
        }
        return {basisX.z, basisY.z, basisZ.z};
    }
    void SetRow(GmAxis axis, const GmVec3 &row) {
        if (axis == GmAxis::X) {
            basisX.x = row.x;
            basisY.x = row.y;
            basisZ.x = row.z;
        } else if (axis == GmAxis::Y) {
            basisX.y = row.x;
            basisY.y = row.y;
            basisZ.y = row.z;
        } else {
            basisX.z = row.x;
            basisY.z = row.y;
            basisZ.z = row.z;
        }
    }
    float &Element(GmAxis row, GmAxis column) {
        GmVec3 &basis = Basis(column);
        if (row == GmAxis::X) {
            return basis.x;
        }
        return row == GmAxis::Y ? basis.y : basis.z;
    }
    const float &Element(GmAxis row, GmAxis column) const {
        const GmVec3 &basis = Basis(column);
        if (row == GmAxis::X) {
            return basis.x;
        }
        return row == GmAxis::Y ? basis.y : basis.z;
    }
    void GetLine(unsigned long index, GmVec3 &value) const;
    void SetLine(unsigned long index, const GmVec3 &value);
    void Set(const GmMat3 &rhs);
    void SetIdentity(void);
    void Set(GmQuat quat);
    void Mult(float scale);
    void Transpose(void);
    void SetTranspose(const GmMat3 &rhs);
    void SetMult(const GmMat3 &lhs, const GmMat3 &rhs);
    void SetBlend(const GmMat3 &from, const GmMat3 &to, float blend);
    void MultTranspose(const GmMat3 &rhs);
    void RotateX(float angle);
    void RotateY(float angle);
    void RotateZ(float angle);
    void SetRotateQuarterY(unsigned long quarterTurn);
    void SetUpVandDOV(const GmVec3 &up, const GmVec3 &directionOfView);
    unsigned long IsNearlyEqual(const GmMat3 &rhs) const;
    unsigned long Inverse(void);
    unsigned long IsOrthogonal(void) const;
    unsigned long IsIndirect(void) const;
    unsigned long IsOrthonormal(void) const;
    void OrthoNormalize(void);
    void Mult(const GmMat3 &rhs);
    void LeftMult(const GmMat3 &lhs);
    static GmMat3 Zero(void) {
        return {};
    }
    void SetDiagonal(float x, float y, float z) {
        *this = Zero();
        basisX.x = x;
        basisY.y = y;
        basisZ.z = z;
    }
};

struct SDynaMath {
    static void ComputeImpulse(
            float mass,
            const GmMat3 &inertia,
            float restitution,
            const GmVec3 &speed,
            const GmVec3 &normal,
            const GmVec3 &lever,
            GmVec3 &out);
};

struct GmIso4 {
    GmMat3 rotation{};
    GmVec3 translation{};

    float &Element(GmAxis row, GmAxis column) {
        return rotation.Element(row, column);
    }
    const float &Element(GmAxis row, GmAxis column) const {
        return rotation.Element(row, column);
    }
    void GetDir(GmVec3 &out) const;
    void SetTranslation(const GmVec3 &translation);
    void SetIdentity(void);
    void Inverse(void);
    void SetUScaleTrans(float scale, const GmVec3 &translation);
    void SetNUScaleTrans(const GmVec3 &scale, const GmVec3 &translation);
    void SetInverse(const GmIso4 &rhs);
    void UScaleSetInverse(const GmIso4 &rhs);
    void NUScaleSetInverse(const GmIso4 &rhs);
    void SetMult(const GmIso4 &lhs, const GmIso4 &rhs);
    void SetBlend(const GmIso4 &from, const GmIso4 &to, float blend);
    void Mult(const GmIso4 &rhs);
    void MultInverse(const GmIso4 &rhs);
    void LeftMult(const GmIso4 &lhs);
    void RotateX(float angle);
    void RotateY(float angle);
    void RotateZ(float angle);
    GmMat3 RotationMatrix() const {
        return rotation;
    }
    GmVec3 Translation() const {
        return translation;
    }
    void Set(const GmMat3 &rotation, const GmVec3 &translation);
    unsigned long IsNearlyEqual(const GmIso4 &rhs) const;
    GmVec3 TranslationForGmSurf(void) const;
    GmVec3 TransformPointForGmSurf(const GmVec3 &point) const;
    GmVec3 SetMultPointForGmSurf(const GmVec3 &point) const;
    GmVec3 InverseRotateTranslatedPointForGmSurf(const GmVec3 &point) const;
    void ScaleRowsForGmSurf(const GmVec3 &rowScale);
};

struct GmBoxAligned {
    GmVec3 center;
    GmVec3 halfExtents;
    static GmBoxAligned FromCenterHalfExtents(const GmVec3 &center,
                                              const GmVec3 &halfExtents);
    GmVec3 Center(void) const;
    GmVec3 HalfExtents(void) const;
    int OverlapsOrientedRelativeTo(const GmBoxAligned &rhs,
                                   const GmIso4 &rhsToSelf,
                                   const GmMat3 &absRhsToSelfRotation) const;
    static int SatAxisOverlapsForGmSurf(float distance, float radius);
    static float ClampScalarForGmSurf(float value, float minValue, float maxValue);
    int TriangleAxisOverlapsForGmSurf(const GmVec3 &triangleVertex0,
                                      const GmVec3 &triangleVertex1,
                                      const GmVec3 &triangleVertex2,
                                      const GmVec3 &axis) const;
    int TrianglePlaneOverlapsForGmSurf(const GmVec3 &triangleVertex0,
                                       const GmVec3 &normal) const;
    int OverlapsTriangleInLocalSpaceForGmSurf(const GmVec3 &triangleVertex0,
                                              const GmVec3 &triangleVertex1,
                                              const GmVec3 &triangleVertex2) const;
    int TestInter(const GmBoxAligned &rhs) const;
    bool TestInterSegment(const GmVec3 &start, const GmVec3 &end) const;
    bool TestInterSegmentMiddleVector(const GmVec3 &middle,
                                     const GmVec3 &delta) const;
    void SetMinMax(const GmVec3 &minPoint, const GmVec3 &maxPoint);
    void GetDiag(GmVec3 &out) const;
    void SetMult(const GmBoxAligned &box, const GmIso4 &iso);
    void Mult(const GmIso4 &iso);
    void Union(const GmBoxAligned &other);
    void SetInvalidForPlugTree(void);
    int IsValidForPlugTreeRefresh(void) const;
    void AddValidPlugTreeBox(const GmBoxAligned &other);
    GmAxis LongestAxis(void) const;
};

struct GmRectAligned {
    float minX;
    float minY;
    float maxX;
    float maxY;
    unsigned long IsInside(const GmVec2 &point) const;
};
