#pragma once

#include <vector>

#include "engine/core/cmw_nod.h"
#include "engine/core/mw_id.h"
struct OptimizedCpuModel3VehicleForceAccess;

class CFunc : public CMwNod {
public:
    CFunc(void);
    ~CFunc(void) override;
};

class CFuncKeys : public CFunc {
public:
    CFuncKeys(void);
    ~CFuncKeys(void) override;
    void GetPreviousKey(float x, unsigned long &keyIndex) const;
    void GetBoundingIndices(float x,
                            unsigned long &keyIndex,
                            unsigned long &nextKeyIndex,
                            int searchForward) const;
    int ComputeBlendCoef(float x,
                         unsigned long &keyIndex,
                         unsigned long &nextKeyIndex,
                         float &blendCoef,
                         int searchForward) const;
    virtual void RemoveKey(unsigned long index);
    virtual void Reset(void);
    virtual void AddKey(float x);
    unsigned long InsertKeyX(float x);
    virtual const CMwId *MwGetId(void) const;
    unsigned long KeyCount(void) const;
    float XAt(unsigned long index) const;

protected:
    friend struct OptimizedCpuModel3VehicleForceAccess;

    std::vector<float> keyPositions;
    CMwId id;
};

class CFuncKeysReal : public CFuncKeys {
public:
    enum ERealInterp {
        Linear,
        Constant,
    };

    struct Key {
        float x = 0.0f;
        float value = 0.0f;
    };

    CFuncKeysReal(void);
    ~CFuncKeysReal(void) override;
    void GetRealAt(float x,
                   float &out,
                   unsigned long &keyIndex,
                   unsigned long &nextKeyIndex,
                   float &blendCoef,
                   ERealInterp interpolation,
                   int searchForward) const;
    void GetValue(float x, float &out, unsigned long &keyIndex) const;
    float GetValue(float x, unsigned long *keyIndex) const;
    void RemoveKey(unsigned long index) override;
    void AddKeyReal(float x, float value);
    unsigned long InsertKeyReal(float x, float value);
    void Reset(void) override;
    void AddKey(float x) override;
    float ValueAt(unsigned long index) const;
    ERealInterp Interpolation(void) const;
    void SetInterpolation(ERealInterp interpolation);
    void SetKeys(std::vector<Key> keys, ERealInterp interpolation);

private:
    friend struct OptimizedCpuModel3VehicleForceAccess;

    std::vector<float> values;
    ERealInterp interpolationMode = Linear;
};
