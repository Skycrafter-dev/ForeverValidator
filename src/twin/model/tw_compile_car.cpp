// CarPack compile from src/twin/model/stadiumcar-values.json. See tw_compile_car.h.
#include "tw_compile_car.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace tw {
namespace model {
namespace {

// ---- minimal JSON reader for the dump format ------------------------------------
// {"meta": {...}, "values": {"path": {"type": T, "value": V, "bits": "0x..."}, ...}}
struct Entry {
    std::string type;
    std::string value;   // raw token: number text, true/false, or string contents
    std::string bits;
};

class Reader {
public:
    explicit Reader(const std::string &text) : s_(text) {}
    bool Parse(std::map<std::string, Entry> *out, std::string *error) {
        if (!Expect('{')) return Fail(error, "top-level object");
        while (true) {
            std::string key;
            if (!String(&key)) return Fail(error, "top-level key");
            if (!Expect(':')) return Fail(error, "':'");
            if (key == "values") {
                if (!Values(out)) return Fail(error, "values object");
            } else if (!Skip()) {
                return Fail(error, "value of " + key);
            }
            if (Peek() == ',') { ++p_; continue; }
            if (Expect('}')) return true;
            return Fail(error, "'}'");
        }
    }

private:
    bool Values(std::map<std::string, Entry> *out) {
        if (!Expect('{')) return false;
        if (Peek() == '}') { ++p_; return true; }
        while (true) {
            std::string path;
            if (!String(&path) || !Expect(':') || !Expect('{')) return false;
            Entry e;
            while (true) {
                std::string k;
                if (!String(&k) || !Expect(':')) return false;
                std::string v;
                if (Peek() == '"') { if (!String(&v)) return false; }
                else if (!Scalar(&v)) return false;
                if (k == "type") e.type = v;
                else if (k == "value") e.value = v;
                else if (k == "bits") e.bits = v;
                if (Peek() == ',') { ++p_; continue; }
                if (!Expect('}')) return false;
                break;
            }
            (*out)[path] = e;
            if (Peek() == ',') { ++p_; continue; }
            return Expect('}');
        }
    }
    bool Skip() {
        const char c = Peek();
        if (c == '"') { std::string t; return String(&t); }
        if (c == '{' || c == '[') {
            const char open = c, close = c == '{' ? '}' : ']';
            ++p_;
            int depth = 1;
            while (p_ < s_.size() && depth > 0) {
                const char d = s_[p_];
                if (d == '"') { std::string t; if (!String(&t)) return false; continue; }
                if (d == open) ++depth;
                else if (d == close) --depth;
                ++p_;
            }
            return depth == 0;
        }
        std::string t;
        return Scalar(&t);
    }
    bool Scalar(std::string *out) {
        Ws();
        const std::size_t b = p_;
        while (p_ < s_.size() && std::strchr(",}] \t\r\n", s_[p_]) == nullptr) ++p_;
        *out = s_.substr(b, p_ - b);
        return !out->empty();
    }
    bool String(std::string *out) {
        Ws();
        if (p_ >= s_.size() || s_[p_] != '"') return false;
        ++p_;
        out->clear();
        while (p_ < s_.size() && s_[p_] != '"') {
            if (s_[p_] == '\\' && p_ + 1 < s_.size()) ++p_;
            out->push_back(s_[p_++]);
        }
        if (p_ >= s_.size()) return false;
        ++p_;
        return true;
    }
    char Peek() { Ws(); return p_ < s_.size() ? s_[p_] : '\0'; }
    bool Expect(char c) { if (Peek() != c) return false; ++p_; return true; }
    void Ws() { while (p_ < s_.size() && std::strchr(" \t\r\n", s_[p_]) != nullptr) ++p_; }
    static bool Fail(std::string *error, const std::string &what) { *error = "json: expected " + what; return false; }
    const std::string &s_;
    std::size_t p_ = 0;
};

// ---- typed access ------------------------------------------------------------------
class Values {
public:
    explicit Values(const std::map<std::string, Entry> &m) : m_(m) {}
    bool Has(const std::string &path) const { return m_.count(path) != 0u; }
    bool F32(const std::string &path, float *out) {
        std::uint32_t u;
        if (!Bits(path, &u)) return false;
        std::memcpy(out, &u, 4);
        return true;
    }
    bool Bits(const std::string &path, std::uint32_t *out) {
        const Entry *e = Find(path, "f32");
        if (e == nullptr) return false;
        if (e->bits.size() < 3 || e->bits.compare(0, 2, "0x") != 0) return Bad(path, "bits");
        *out = static_cast<std::uint32_t>(std::strtoul(e->bits.c_str() + 2, nullptr, 16));
        return true;
    }
    bool U32(const std::string &path, std::uint32_t *out) {
        const Entry *e = Find(path, "u32");
        if (e == nullptr) return false;
        char *end = nullptr;
        const unsigned long v = std::strtoul(e->value.c_str(), &end, 10);
        if (end == e->value.c_str() || *end != '\0') return Bad(path, "u32 value");
        *out = static_cast<std::uint32_t>(v);
        return true;
    }
    bool I32(const std::string &path, std::int32_t *out) {
        const Entry *e = Find(path, "i32");
        if (e == nullptr) return false;
        *out = static_cast<std::int32_t>(std::strtol(e->value.c_str(), nullptr, 10));
        return true;
    }
    bool Bool(const std::string &path, std::uint32_t *out) {
        const Entry *e = Find(path, "bool");
        if (e == nullptr) return false;
        if (e->value == "true") *out = 1u;
        else if (e->value == "false") *out = 0u;
        else return Bad(path, "bool value");
        return true;
    }
    bool Text(const std::string &path, const char *type, std::string *out) {
        const Entry *e = Find(path, type);
        if (e == nullptr) return false;
        *out = e->value;
        return true;
    }
    const std::string &Error() const { return error_; }

private:
    const Entry *Find(const std::string &path, const char *type) {
        auto it = m_.find(path);
        if (it == m_.end()) { error_ = "missing " + path; return nullptr; }
        if (it->second.type != type) { error_ = path + ": type " + it->second.type + ", expected " + type; return nullptr; }
        return &it->second;
    }
    bool Bad(const std::string &path, const char *what) { error_ = path + ": bad " + what; return false; }
    const std::map<std::string, Entry> &m_;
    std::string error_;
};

const char *kCurveJsonName[] = {
#define TW_CURVE(name, json) json,
#include "curve_ids.inc"
#undef TW_CURVE
};

const char *kVec[3] = {"x", "y", "z"};
const char *kBasis[3] = {"basisX", "basisY", "basisZ"};

std::string Idx(const char *base, int i) { return std::string(base) + "[" + std::to_string(i) + "]"; }

}  // namespace

bool CompileCarFromJson(const std::string &path, TwCarPack *pack, std::string *error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { *error = "cannot open " + path; return false; }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::map<std::string, Entry> entries;
    Reader reader(text);
    if (!reader.Parse(&entries, error)) return false;
    Values v(entries);
    *pack = TwCarPack{};
#define TW_REQ(expr) do { if (!(expr)) { *error = v.Error().empty() ? std::string("check failed: ") + #expr : v.Error(); return false; } } while (0)

    // scalars
#define TW_CAR(member, type, p) TW_REQ(TW_CAR_READ_##type(p, &pack->member));
#define TW_CAR_READ_F32(p, out) v.F32(p, out)
#define TW_CAR_READ_U32(p, out) v.U32(p, out)
#define TW_CAR_READ_I32(p, out) v.I32(p, out)
#define TW_CAR_READ_BOOL(p, out) v.Bool(p, out)
#include "car_fields.inc"
#undef TW_CAR

    // structure the twin specialises on
    TW_REQ(pack->handlingModel == 5u);
    TW_REQ(pack->wheelForceMode == 2u);

    // transmission arrays
    const char *base = "tuning.gearedDrive.transmission.";
    struct Arr { const char *name; float *dst; std::uint32_t *size; } arrays[] = {
        {"gearSpeedRatio", pack->gearSpeedRatio, &pack->gearCount},
        {"upshiftThreshold", pack->upshiftThreshold, nullptr},
        {"downshiftThreshold", pack->downshiftThreshold, nullptr},
        {"targetInputBias", pack->targetInputBias, nullptr},
        {"rpmDelta", pack->rpmDelta, nullptr},
        {"rpmWanted", pack->rpmWanted, &pack->rpmWantedCount},
    };
    for (const Arr &a : arrays) {
        std::uint32_t n = 0;
        TW_REQ(v.U32(std::string(base) + a.name + ".size", &n));
        TW_REQ(n <= static_cast<std::uint32_t>(kTwMaxGears));
        if (a.size != nullptr) *a.size = n;
        else if (n != pack->gearCount) { *error = std::string(a.name) + ": size differs from gearSpeedRatio"; return false; }
        for (std::uint32_t i = 0; i < n; ++i) TW_REQ(v.F32(Idx((std::string(base) + a.name).c_str(), static_cast<int>(i)), &a.dst[i]));
    }

    // curves
    for (std::uint32_t c = 0; c < kTwCurveCount; ++c) {
        const std::string cb = std::string("tuning.curves.") + kCurveJsonName[c] + ".";
        std::uint32_t present = 0, interp = 0, count = 0;
        TW_REQ(v.Bool(cb + "present", &present));
        TW_REQ(v.U32(cb + "interpolation", &interp));
        TW_REQ(v.U32(cb + "keyCount", &count));
        TW_REQ(interp <= 1u && count <= static_cast<std::uint32_t>(kTwCurveMaxKeys));
        float x[kTwCurveMaxKeys], y[kTwCurveMaxKeys];
        for (std::uint32_t k = 0; k < count; ++k) {
            TW_REQ(v.F32(Idx((cb + "keys").c_str(), static_cast<int>(k)) + ".position", &x[k]));
            TW_REQ(v.F32(Idx((cb + "keys").c_str(), static_cast<int>(k)) + ".value", &y[k]));
        }
        pack->curvePresent[c] = present;
        TW_REQ(TwCurveCompile(&pack->curve[c], present ? count : 0u, x, y, interp == 1u));
    }

    // wheels (derived.wheelSet = VehicleWheelSetDefinition, board wheel order)
    for (int w = 0; w < 4; ++w) {
        const std::string wb = Idx("derived.wheelSet.wheels", w) + ".";
        TwCarWheel &wh = pack->wheel[w];
        std::uint32_t axle = 0;
        TW_REQ(v.U32(wb + "surfaceId", &wh.surfaceId));
        TW_REQ(v.Bool(wb + "killsLateralSpeedOnContact", &wh.killsLateralSpeedOnContact));
        TW_REQ(v.U32(wb + "axle", &axle));
        wh.axleFront = axle == 1u ? 1u : 0u;
        TW_REQ(v.F32(wb + "rollingRadius", &wh.rollingRadius));
        for (int k = 0; k < 3; ++k) {
            TW_REQ(v.F32(wb + "forceApplicationPoint." + kVec[k], &wh.forceApplicationPoint[k]));
            TW_REQ(v.F32(wb + "restSurfacePose.translation." + kVec[k], &wh.restPoint[k]));
        }
    }

    // collision shapes (tree order)
    for (int s = 0; s < 8; ++s) {
        const std::string sb = Idx("solid.collisionShapes", s) + ".";
        TwCarShape &sh = pack->shape[s];
        TW_REQ(v.U32(sb + "role", &sh.role));
        TW_REQ(v.U32(sb + "surfaceType", &sh.surfaceType));
        TW_REQ(v.U32(sb + "surfaceMaterial", &sh.surfaceMaterial));
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) TW_REQ(v.F32(sb + "localPose.rotation." + kBasis[c] + "." + kVec[k], &sh.rotation[3 * c + k]));
        for (int k = 0; k < 3; ++k) {
            TW_REQ(v.F32(sb + "localPose.translation." + kVec[k], &sh.translation[k]));
            TW_REQ(v.F32(sb + "localBounds.center." + kVec[k], &sh.center[k]));
            TW_REQ(v.F32(sb + "localBounds.halfExtents." + kVec[k], &sh.radii[k]));
        }
        // wheel shapes: translation equals a wheel rest point bit for bit
        sh.wheelIndex = -1;
        for (int w = 0; w < 4; ++w)
            if (std::memcmp(sh.translation, pack->wheel[w].restPoint, sizeof(sh.translation)) == 0) sh.wheelIndex = w;
    }
    {
        int wheelShapes = 0;
        for (const TwCarShape &sh : pack->shape) wheelShapes += sh.wheelIndex >= 0;
        TW_REQ(wheelShapes == 4);
    }

    // materials
    pack->materialCount = 0;
    while (pack->materialCount < static_cast<std::uint32_t>(kTwMaxMaterials) &&
           v.Has(Idx("vehicle.materials", static_cast<int>(pack->materialCount)) + ".naturalId")) {
        const std::string mb = Idx("vehicle.materials", static_cast<int>(pack->materialCount)) + ".";
        TwCarMaterial &m = pack->material[pack->materialCount];
        TW_REQ(v.U32(mb + "naturalId", &m.naturalId));
        const char *bxyz[4] = {"x", "y", "z", "w"};
        for (int k = 0; k < 4; ++k) TW_REQ(v.F32(mb + "blend." + bxyz[k], &m.blend[k]));
        TW_REQ(v.F32(mb + "fakeContactPeriodX", &m.fakeContactPeriodX));
        TW_REQ(v.F32(mb + "fakeContactPeriodZ", &m.fakeContactPeriodZ));
        TW_REQ(v.F32(mb + "fakeContactSpeedScale", &m.fakeContactSpeedScale));
        TW_REQ(v.F32(mb + "fakeContactDepthMax", &m.fakeContactDepthMax));
        TW_REQ(v.F32(mb + "feedbackSpeedDivisor", &m.feedbackSpeedDivisor));
        TW_REQ(v.F32(mb + "feedbackScale", &m.feedbackScale));
        TW_REQ(v.Bool(mb + "usesFakeContactTexture", &m.usesFakeContactTexture));
        ++pack->materialCount;
    }
    TW_REQ(pack->materialCount > 0u);
    for (int id = 0; id < kTwNaturalIds; ++id) {
        const std::string p = Idx("vehicle.materialIndexByNaturalId", id);
        if (!v.Has(p)) { pack->materialIndexByNaturalId[id] = 0u; continue; }
        TW_REQ(v.U32(p, &pack->materialIndexByNaturalId[id]));
        TW_REQ(pack->materialIndexByNaturalId[id] < pack->materialCount);
    }

    // fake-contact texture, red channel
    {
        std::uint32_t w = 0, h = 0;
        std::string hex;
        TW_REQ(v.U32("vehicle.fakeContactTexture.width", &w));
        TW_REQ(v.U32("vehicle.fakeContactTexture.height", &h));
        TW_REQ(w == static_cast<std::uint32_t>(kTwFakeTextureSize) && h == static_cast<std::uint32_t>(kTwFakeTextureSize));
        TW_REQ(v.Text("vehicle.fakeContactTexture.redChannelHexRowMajor", "hex", &hex));
        TW_REQ(hex.size() == 2u * w * h);
        for (std::size_t i = 0; i < w * h; ++i) {
            const char b[3] = {hex[2 * i], hex[2 * i + 1], '\0'};
            pack->fakeTextureRed[i] = static_cast<std::uint8_t>(std::strtoul(b, nullptr, 16));
        }
    }

    // folded constants
    TW_REQ(pack->derived_dyna_mass > 0.0f);
    pack->invMass = 1.0f / pack->derived_dyna_mass;
    pack->invInertiaDiag[0] = pack->derived_dyna_inverseBodyInertia_basisX_x;
    pack->invInertiaDiag[1] = pack->derived_dyna_inverseBodyInertia_basisY_y;
    pack->invInertiaDiag[2] = pack->derived_dyna_inverseBodyInertia_basisZ_z;
    TW_REQ(pack->derived_dyna_inverseBodyInertia_basisX_y == 0.0f && pack->derived_dyna_inverseBodyInertia_basisX_z == 0.0f &&
           pack->derived_dyna_inverseBodyInertia_basisY_x == 0.0f && pack->derived_dyna_inverseBodyInertia_basisY_z == 0.0f &&
           pack->derived_dyna_inverseBodyInertia_basisZ_x == 0.0f && pack->derived_dyna_inverseBodyInertia_basisZ_y == 0.0f);
    pack->com[0] = pack->derived_dyna_localCenterOfMass_x;
    pack->com[1] = pack->derived_dyna_localCenterOfMass_y;
    pack->com[2] = pack->derived_dyna_localCenterOfMass_z;
    {
        float zmin = pack->wheel[0].restPoint[2], zmax = zmin;
        for (int w = 1; w < 4; ++w) {
            if (pack->wheel[w].restPoint[2] < zmin) zmin = pack->wheel[w].restPoint[2];
            if (zmax < pack->wheel[w].restPoint[2]) zmax = pack->wheel[w].restPoint[2];
        }
        const float half = (zmax - zmin) * 0.5f;   // GmBoxAligned::SetMinMax, then half + half (WH:857-858)
        pack->wheelLongitudinalSpan = half + half;
    }
    pack->maxAngularSpeed = pack->contactResponse_pointImpulseAngularSpeedMax;
    pack->gravityY = -10.0f;
    pack->zoneLinearDamping = 1.0f;
    pack->zoneAngularDamping = 1.0f;
    std::memcpy(&pack->fsGroundedBits, &pack->bodyAirResponse_groundedSolidFeedback1, 4);
    pack->ldsGroundedBits = 0u;
    std::memcpy(&pack->fsAirborneBits, &pack->bodyAirResponse_airborneSolidFeedback1, 4);
    std::memcpy(&pack->ldsAirborneBits, &pack->bodyAirResponse_airborneSolidFeedback0, 4);
#undef TW_REQ
    return true;
}

bool ReadCarJsonValues(const std::string &path, std::vector<std::pair<std::string, TwJsonValue>> *out, std::string *error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { *error = "cannot open " + path; return false; }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::map<std::string, Entry> entries;
    Reader reader(text);
    if (!reader.Parse(&entries, error)) return false;
    out->clear();
    for (const auto &kv : entries) {
        TwJsonValue v;
        v.type = kv.second.type;
        if (v.type == "f32") v.word = static_cast<std::uint32_t>(std::strtoul(kv.second.bits.c_str() + 2, nullptr, 16));
        else if (v.type == "u32") v.word = static_cast<std::uint32_t>(std::strtoul(kv.second.value.c_str(), nullptr, 10));
        else if (v.type == "i32") v.word = static_cast<std::uint32_t>(std::strtol(kv.second.value.c_str(), nullptr, 10));
        else if (v.type == "bool") v.word = kv.second.value == "true" ? 1u : 0u;
        else v.text = kv.second.value;
        out->push_back({kv.first, v});
    }
    return true;
}

std::vector<TwPackValue> ExportCarPackValues(const TwCarPack &pack) {
    std::vector<TwPackValue> out;
    auto f = [&](const std::string &p, float x) { std::uint32_t u; std::memcpy(&u, &x, 4); out.push_back({p, u, {}}); };
    auto u = [&](const std::string &p, std::uint32_t x) { out.push_back({p, x, {}}); };
#define TW_CAR(member, type, p) TW_CAR_EXPORT_##type(p, pack.member);
#define TW_CAR_EXPORT_F32(p, x) f(p, x)
#define TW_CAR_EXPORT_U32(p, x) u(p, x)
#define TW_CAR_EXPORT_I32(p, x) u(p, static_cast<std::uint32_t>(x))
#define TW_CAR_EXPORT_BOOL(p, x) u(p, x)
#include "car_fields.inc"
#undef TW_CAR
    const std::string base = "tuning.gearedDrive.transmission.";
    struct Arr { const char *name; const float *src; std::uint32_t n; } arrays[] = {
        {"gearSpeedRatio", pack.gearSpeedRatio, pack.gearCount}, {"upshiftThreshold", pack.upshiftThreshold, pack.gearCount},
        {"downshiftThreshold", pack.downshiftThreshold, pack.gearCount}, {"targetInputBias", pack.targetInputBias, pack.gearCount},
        {"rpmDelta", pack.rpmDelta, pack.gearCount}, {"rpmWanted", pack.rpmWanted, pack.rpmWantedCount}};
    for (const Arr &a : arrays) {
        u(base + a.name + ".size", a.n);
        for (std::uint32_t i = 0; i < a.n; ++i) f(Idx((base + a.name).c_str(), static_cast<int>(i)), a.src[i]);
    }
    for (std::uint32_t c = 0; c < kTwCurveCount; ++c) {
        const std::string cb = std::string("tuning.curves.") + kCurveJsonName[c] + ".";
        const TwCurve &cv = pack.curve[c];
        u(cb + "present", pack.curvePresent[c]);
        u(cb + "interpolation", cv.linear);
        u(cb + "keyCount", cv.count);
        for (std::uint32_t k = 0; k < cv.count; ++k) {
            f(Idx((cb + "keys").c_str(), static_cast<int>(k)) + ".position", cv.x[k]);
            f(Idx((cb + "keys").c_str(), static_cast<int>(k)) + ".value", cv.v[k]);
        }
    }
    for (int w = 0; w < 4; ++w) {
        const std::string wb = Idx("derived.wheelSet.wheels", w) + ".";
        const TwCarWheel &wh = pack.wheel[w];
        u(wb + "surfaceId", wh.surfaceId);
        u(wb + "killsLateralSpeedOnContact", wh.killsLateralSpeedOnContact);
        u(wb + "axle", wh.axleFront);
        f(wb + "rollingRadius", wh.rollingRadius);
        for (int k = 0; k < 3; ++k) {
            f(wb + "forceApplicationPoint." + kVec[k], wh.forceApplicationPoint[k]);
            f(wb + "restSurfacePose.translation." + kVec[k], wh.restPoint[k]);
        }
    }
    for (int s = 0; s < 8; ++s) {
        const std::string sb = Idx("solid.collisionShapes", s) + ".";
        const TwCarShape &sh = pack.shape[s];
        u(sb + "role", sh.role);
        u(sb + "surfaceType", sh.surfaceType);
        u(sb + "surfaceMaterial", sh.surfaceMaterial);
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) f(sb + "localPose.rotation." + kBasis[c] + "." + kVec[k], sh.rotation[3 * c + k]);
        for (int k = 0; k < 3; ++k) {
            f(sb + "localPose.translation." + kVec[k], sh.translation[k]);
            f(sb + "localBounds.center." + kVec[k], sh.center[k]);
            f(sb + "localBounds.halfExtents." + kVec[k], sh.radii[k]);
        }
    }
    for (std::uint32_t i = 0; i < pack.materialCount; ++i) {
        const std::string mb = Idx("vehicle.materials", static_cast<int>(i)) + ".";
        const TwCarMaterial &m = pack.material[i];
        const char *bxyz[4] = {"x", "y", "z", "w"};
        u(mb + "naturalId", m.naturalId);
        for (int k = 0; k < 4; ++k) f(mb + "blend." + bxyz[k], m.blend[k]);
        f(mb + "fakeContactPeriodX", m.fakeContactPeriodX);
        f(mb + "fakeContactPeriodZ", m.fakeContactPeriodZ);
        f(mb + "fakeContactSpeedScale", m.fakeContactSpeedScale);
        f(mb + "fakeContactDepthMax", m.fakeContactDepthMax);
        f(mb + "feedbackSpeedDivisor", m.feedbackSpeedDivisor);
        f(mb + "feedbackScale", m.feedbackScale);
        u(mb + "usesFakeContactTexture", m.usesFakeContactTexture);
    }
    for (int id = 0; id < 31; ++id) u(Idx("vehicle.materialIndexByNaturalId", id), pack.materialIndexByNaturalId[id]);
    u("vehicle.fakeContactTexture.width", kTwFakeTextureSize);
    u("vehicle.fakeContactTexture.height", kTwFakeTextureSize);
    {
        static const char *hexd = "0123456789abcdef";
        std::string hex;
        hex.reserve(2u * sizeof(pack.fakeTextureRed));
        for (std::uint8_t b : pack.fakeTextureRed) { hex.push_back(hexd[b >> 4]); hex.push_back(hexd[b & 15]); }
        out.push_back({"vehicle.fakeContactTexture.redChannelHexRowMajor", 0u, hex});
    }
    return out;
}

}  // namespace model
}  // namespace tw
