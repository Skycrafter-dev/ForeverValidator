// Twin substep policies: names (policy.h).
#include "policy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tw {
namespace step {

namespace {

// Value after the first occurrence of `key` at or after `from`, or `fallback`.
bool Number(const std::string &s, std::size_t from, char key, double fallback, double *out, std::string *error) {
    const std::size_t at = s.find(key, from);
    if (at == std::string::npos) {
        *out = fallback;
        return true;
    }
    // digits and one '.', no exponent: "d2e3" is d = 2 then e = 3 (strtod would read 2000)
    const char *begin = s.c_str() + at + 1;
    const char *end = begin;
    double value = 0.0, scale = 0.0;
    for (; (*end >= '0' && *end <= '9') || (*end == '.' && scale == 0.0); ++end) {
        if (*end == '.') {
            scale = 1.0;
            continue;
        }
        if (scale == 0.0) {
            value = value * 10.0 + (*end - '0');
        } else {
            scale *= 0.1;
            value += (*end - '0') * scale;
        }
    }
    *out = value;
    if (end == begin) {
        if (error) *error = std::string("policy '") + s + "': no number after '" + key + "'";
        return false;
    }
    return true;
}

std::string Num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

}  // namespace

bool ParsePolicy(const char *text, PolicyConfig *out, std::string *error) {
    if (text == nullptr || *text == '\0' || !std::strcmp(text, "fv")) {
        *out = PolicyConfig::Fv();
        return true;
    }
    const std::string s(text);
    PolicyConfig p;
    std::size_t head = 0;
    if (s.rfind("cap", 0) == 0) {
        p.kind = PolicyKind::Cap;
        head = 3;
    } else if (s.rfind("contact", 0) == 0) {
        p.kind = PolicyKind::ContactAware;
        head = 7;
    } else if (s.rfind("spec", 0) == 0) {
        p.kind = PolicyKind::Speculative;
        head = 4;
    } else {
        if (error) *error = "unknown policy '" + s + "' (fv | capK[vV] | contactKdD[eE][wW][vV][nN][pP][h0][m1][sS] | specK...)";
        return false;
    }
    char *end = nullptr;
    const long k = std::strtol(s.c_str() + head, &end, 10);
    if (end == s.c_str() + head || k < 1) {
        if (error) *error = "policy '" + s + "': K must be >= 1";
        return false;
    }
    p.k = static_cast<std::uint32_t>(k);
    for (const char *c = end; *c != '\0'; ++c) {
        const bool allowed = p.kind == PolicyKind::Cap ? std::strchr("v0123456789.", *c) != nullptr
                                                       : std::strchr(p.kind == PolicyKind::Speculative ? "ewvnpuarbfx0123456789."
                                                                                                       : "dewvnphmsuarbfx0123456789.",
                                                                     *c) != nullptr;
        if (!allowed) {
            if (error) *error = "policy '" + s + "': unexpected '" + std::string(1, *c) + "'";
            return false;
        }
    }
    const std::size_t from = static_cast<std::size_t>(end - s.c_str());
    double v = 0.0;
    if (!Number(s, from, 'v', 1e30, &v, error)) return false;
    p.fvAboveKmh = static_cast<float>(v);
    if (p.kind == PolicyKind::ContactAware || p.kind == PolicyKind::Speculative) {
        if (!Number(s, from, 'd', 0.0, &v, error)) return false;
        p.lookahead = static_cast<float>(v);
        if (!Number(s, from, 'e', static_cast<double>(kMaxSubsteps), &v, error)) return false;
        p.eventCap = static_cast<std::uint32_t>(v);
        if (!Number(s, from, 'w', 1e30, &v, error)) return false;
        p.eventFullKmh = static_cast<float>(v);
        if (!Number(s, from, 'n', 1e30, &v, error)) return false;
        p.normalSpeed = static_cast<float>(v);
        if (!Number(s, from, 'p', 1e30, &v, error)) return false;
        p.depth = static_cast<float>(v);
        if (!Number(s, from, 'h', 1.0, &v, error)) return false;
        p.probeHalf = v != 0.0;
        if (!Number(s, from, 'm', 0.0, &v, error)) return false;
        p.probeMidOnly = v != 0.0;
        if (!Number(s, from, 's', 0.0, &v, error)) return false;
        p.lookStep = static_cast<float>(v);
        if (!Number(s, from, 'u', 1e30, &v, error)) return false;
        p.twoAboveKmh = static_cast<float>(v);
        if (!Number(s, from, 'a', 0.0, &v, error)) return false;
        p.twoAirborneOnly = v != 0.0;
        if (!Number(s, from, 'r', 1e30, &v, error)) return false;
        p.twoRotation = static_cast<float>(v);
        if (!Number(s, from, 'b', 0.0, &v, error)) return false;
        p.ballistic = v != 0.0;
        if (!Number(s, from, 'f', 1e30, &v, error)) return false;
        p.fullRotation = static_cast<float>(v);
        if (!Number(s, from, 'x', 0.0, &v, error)) return false;
        p.subcycle = v != 0.0;
    }
    *out = p;
    return true;
}

std::string PolicyName(const PolicyConfig &p) {
    switch (p.kind) {
    case PolicyKind::Fv: return "fv";
    case PolicyKind::Cap: {
        std::string s = "cap" + std::to_string(p.k);
        if (p.fvAboveKmh < 1e29f) s += "v" + Num(p.fvAboveKmh);
        return s;
    }
    case PolicyKind::ContactAware: {
        std::string s = "contact" + std::to_string(p.k) + "d" + Num(p.lookahead);
        if (p.eventCap < kMaxSubsteps) s += "e" + std::to_string(p.eventCap);
        if (p.eventFullKmh < 1e29f) s += "w" + Num(p.eventFullKmh);
        if (p.fvAboveKmh < 1e29f) s += "v" + Num(p.fvAboveKmh);
        if (p.normalSpeed < 1e29f) s += "n" + Num(p.normalSpeed);
        if (p.depth < 1e29f) s += "p" + Num(p.depth);
        if (!p.probeHalf) s += "h0";
        if (p.probeMidOnly) s += "m1";
        if (p.lookStep > 0.0f) s += "s" + Num(p.lookStep);
        if (p.twoAboveKmh < 1e29f) s += "u" + Num(p.twoAboveKmh) + (p.twoAirborneOnly ? "a1" : "");
        if (p.twoRotation < 1e29f) s += "r" + Num(p.twoRotation);
        if (p.fullRotation < 1e29f) s += "f" + Num(p.fullRotation);
        if (p.subcycle) s += "x1";
        if (p.ballistic) s += "b1";
        return s;
    }
    case PolicyKind::Speculative: {
        std::string s = "spec" + std::to_string(p.k);
        if (p.eventCap < kMaxSubsteps) s += "e" + std::to_string(p.eventCap);
        if (p.eventFullKmh < 1e29f) s += "w" + Num(p.eventFullKmh);
        if (p.fvAboveKmh < 1e29f) s += "v" + Num(p.fvAboveKmh);
        if (p.normalSpeed < 1e29f) s += "n" + Num(p.normalSpeed);
        if (p.depth < 1e29f) s += "p" + Num(p.depth);
        if (p.twoAboveKmh < 1e29f) s += "u" + Num(p.twoAboveKmh) + (p.twoAirborneOnly ? "a1" : "");
        if (p.twoRotation < 1e29f) s += "r" + Num(p.twoRotation);
        if (p.fullRotation < 1e29f) s += "f" + Num(p.fullRotation);
        if (p.subcycle) s += "x1";
        if (p.ballistic) s += "b1";
        return s;
    }
    }
    return "?";
}

}  // namespace step
}  // namespace tw
