// Lossless TwinState <-> harness golden words (design.md §3, docs/twin/harness.md).
//
// Import reads FV's state as the harness records it (TWINGOLD field names,
// 32-bit words, e.g. the `pre.` / `post.` fields of gate1.tgold) into a
// TwinState; export writes the same words back. For every golden field of
// tiers state and race, export(import(words)) == words bit for bit, with the
// documented non-state fields (state_fields.inc header) reconstructed from
// the context: inputs, map constants, the car pack's fs/lds values.
//
// FV-free and harness-agnostic: the caller supplies
//   find(name)          -> {const std::uint32_t *words, std::uint32_t count}
//                          (words == nullptr if the field is absent)
//   put(name, w, count) -> stores count words for that field.
#ifndef TW_STATE_GOLDEN_H
#define TW_STATE_GOLDEN_H

#include <cstdint>
#include <cstring>
#include <string>

#include "tw_state.h"

namespace tw {
namespace model {

struct TwGoldenWords {
    const std::uint32_t *words = nullptr;
    std::uint32_t count = 0u;
};

// Spawn isos are 12 words (rotation basisX/Y/Z, translation), compared bitwise.
struct TwSpawnTable {
    const std::uint32_t (*iso)[12] = nullptr;
    std::uint32_t count = 0u;
};

struct TwGoldenContext {
    // car pack: force scale / linear damping for grounded and airborne
    std::uint32_t fsGroundedBits = 0u, ldsGroundedBits = 0u;
    std::uint32_t fsAirborneBits = 0u, ldsAirborneBits = 0u;
    // world pack
    TwSpawnTable spawns{};
    std::uint32_t requiredCps = 0u, requiredLaps = 0u;
    // tick inputs echoed in ctl.* (export only; import ignores ctl.gas/brake/steer)
    std::uint32_t ctlGasBits = 0u, ctlBrakeBits = 0u, ctlSteerBits = 0u;
};

enum class TwGoldenStatus : std::uint8_t {
    Ok,
    MissingField,
    BadCount,
    OutOfRange,        // a value the compact field cannot hold
    UnknownFsLds,      // (fs, lds) pair is neither grounded nor airborne
    UnknownSpawn,      // spawn iso not in the spawn table
    HighSlots,         // checkpoint slot >= 64 set
    BadFinish,         // finish estimate not {present, lower, lower+1, lower+1}
    NonZeroAccumulator,
    MapConstantMismatch,
};

inline const char *TwGoldenStatusName(TwGoldenStatus s) {
    switch (s) {
    case TwGoldenStatus::Ok: return "ok";
    case TwGoldenStatus::MissingField: return "missing_field";
    case TwGoldenStatus::BadCount: return "bad_count";
    case TwGoldenStatus::OutOfRange: return "out_of_range";
    case TwGoldenStatus::UnknownFsLds: return "unknown_fs_lds";
    case TwGoldenStatus::UnknownSpawn: return "unknown_spawn";
    case TwGoldenStatus::HighSlots: return "high_slots";
    case TwGoldenStatus::BadFinish: return "bad_finish";
    case TwGoldenStatus::NonZeroAccumulator: return "nonzero_accumulator";
    case TwGoldenStatus::MapConstantMismatch: return "map_constant_mismatch";
    }
    return "?";
}

struct TwGoldenError {
    TwGoldenStatus status = TwGoldenStatus::Ok;
    std::string field;
};

namespace detail {
inline std::uint32_t F2U(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float U2F(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

inline void StoreElement(TwinState *s, const TwFieldInfo &f, int k, std::uint32_t v) {
    unsigned char *p = reinterpret_cast<unsigned char *>(s) + f.offset + k * f.elementBytes;
    switch (f.elementBytes) {
    case 4: std::memcpy(p, &v, 4); break;
    case 2: { std::uint16_t x = static_cast<std::uint16_t>(v); std::memcpy(p, &x, 2); break; }
    default: { std::uint8_t x = static_cast<std::uint8_t>(v); std::memcpy(p, &x, 1); break; }
    }
}
inline std::uint32_t LoadElement(const TwinState &s, const TwFieldInfo &f, int k) {
    const unsigned char *p = reinterpret_cast<const unsigned char *>(&s) + f.offset + k * f.elementBytes;
    switch (f.elementBytes) {
    case 4: { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
    case 2: { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
    default: return *p;
    }
}
inline std::uint32_t NarrowMax(const TwFieldInfo &f) {
    return f.elementBytes == 2 ? 0xFFFFu : (f.type == TwType::I8 ? 0x7Fu : 0xFFu);
}
}  // namespace detail

// golden words -> TwinState
template <class Find>
inline TwGoldenError ImportGolden(Find &&find, const TwGoldenContext &ctx, TwinState *out) {
    TwinStateClear(out);
    std::size_t n = 0;
    const TwFieldInfo *fields = TwinFields(&n);
    auto fail = [](TwGoldenStatus st, const char *name) { return TwGoldenError{st, name}; };
    for (std::size_t i = 0; i < n; ++i) {
        const TwFieldInfo &f = fields[i];
        if (f.codec == TwCodec::NONE) continue;
        const TwGoldenWords g = find(f.golden);
        if (g.words == nullptr) return fail(TwGoldenStatus::MissingField, f.golden);
        switch (f.codec) {
        case TwCodec::DIRECT:
            if (g.count != f.count) return fail(TwGoldenStatus::BadCount, f.golden);
            for (int k = 0; k < f.count; ++k) detail::StoreElement(out, f, k, g.words[k]);
            break;
        case TwCodec::BOOL:
        case TwCodec::NARROW:
            if (g.count != f.count) return fail(TwGoldenStatus::BadCount, f.golden);
            for (int k = 0; k < f.count; ++k) {
                const std::uint32_t v = g.words[k];   // u32, or i32 >= 0
                if (v > (f.codec == TwCodec::BOOL ? 1u : detail::NarrowMax(f))) return fail(TwGoldenStatus::OutOfRange, f.golden);
                detail::StoreElement(out, f, k, v);
            }
            break;
        case TwCodec::PEERZ:
            if (g.count != 3u * f.count) return fail(TwGoldenStatus::BadCount, f.golden);
            for (int k = 0; k < f.count; ++k) detail::StoreElement(out, f, k, g.words[3 * k + 2]);
            break;
        case TwCodec::SIGN: {
            if (g.count != 1u) return fail(TwGoldenStatus::BadCount, f.golden);
            const std::uint32_t b = g.words[0];
            std::int8_t v;
            if (b == 0x00000000u) v = 0;
            else if (b == 0x3f800000u) v = 1;
            else if (b == 0xbf800000u) v = -1;
            else return fail(TwGoldenStatus::OutOfRange, f.golden);
            detail::StoreElement(out, f, 0, static_cast<std::uint8_t>(v));
            break;
        }
        case TwCodec::FSLDS: {
            const TwGoldenWords lds = find("dyna.linear_damping");
            if (g.count != 1u || lds.words == nullptr || lds.count != 1u) return fail(TwGoldenStatus::MissingField, "dyna.linear_damping");
            const std::uint32_t fs = g.words[0], ld = lds.words[0];
            std::uint32_t grounded;
            if (fs == ctx.fsGroundedBits && ld == ctx.ldsGroundedBits) grounded = 1u;
            else if (fs == ctx.fsAirborneBits && ld == ctx.ldsAirborneBits) grounded = 0u;
            else return fail(TwGoldenStatus::UnknownFsLds, f.golden);
            detail::StoreElement(out, f, 0, grounded);
            break;
        }
        case TwCodec::SLOTS:
            if (g.count < f.count) return fail(TwGoldenStatus::BadCount, f.golden);
            for (std::uint32_t k = f.count; k < g.count; ++k)
                if (g.words[k] != 0u) return fail(TwGoldenStatus::HighSlots, f.golden);
            for (int k = 0; k < f.count; ++k) detail::StoreElement(out, f, k, g.words[k]);
            break;
        case TwCodec::FINLO:
        case TwCodec::FINPR: {
            // {present, lower lo/hi, upper lo/hi, estimated lo/hi}; FV's valid estimate has
            // upper = lower + 1 and estimated = upper (finish_time.h IsValid)
            if (g.count != 7u) return fail(TwGoldenStatus::BadCount, f.golden);
            const std::uint64_t lo = g.words[1] | (static_cast<std::uint64_t>(g.words[2]) << 32);
            const std::uint64_t hi = g.words[3] | (static_cast<std::uint64_t>(g.words[4]) << 32);
            const std::uint64_t es = g.words[5] | (static_cast<std::uint64_t>(g.words[6]) << 32);
            if (g.words[0] > 1u) return fail(TwGoldenStatus::BadFinish, f.golden);
            if (g.words[0] == 1u && !(hi == lo + 1u && es == hi)) return fail(TwGoldenStatus::BadFinish, f.golden);
            if (g.words[0] == 0u && (lo | hi | es) != 0u) return fail(TwGoldenStatus::BadFinish, f.golden);
            if (f.codec == TwCodec::FINPR) detail::StoreElement(out, f, 0, g.words[0]);
            else { detail::StoreElement(out, f, 0, g.words[1]); detail::StoreElement(out, f, 1, g.words[2]); }
            break;
        }
        case TwCodec::SPAWN: {
            if (g.count != 12u) return fail(TwGoldenStatus::BadCount, f.golden);
            std::uint32_t idx = ctx.spawns.count;
            for (std::uint32_t s = 0; s < ctx.spawns.count; ++s)
                if (std::memcmp(ctx.spawns.iso[s], g.words, 48) == 0) { idx = s; break; }
            if (idx == ctx.spawns.count) return fail(TwGoldenStatus::UnknownSpawn, f.golden);
            detail::StoreElement(out, f, 0, idx);
            break;
        }
        case TwCodec::NONE: break;
        }
    }
    // documented non-state fields: verify the claims
    const char *zeroFields[] = {"ct.wheel_contacts", "ct.body_contacts", "ct.body_normal_sum"};
    for (const char *z : zeroFields) {
        const TwGoldenWords g = find(z);
        if (g.words == nullptr) return fail(TwGoldenStatus::MissingField, z);
        for (std::uint32_t k = 0; k < g.count; ++k)
            if (g.words[k] != 0u) return fail(TwGoldenStatus::NonZeroAccumulator, z);
    }
    const TwGoldenWords rc = find("race.required_cps"), rl = find("race.required_laps");
    if (rc.words == nullptr || rl.words == nullptr) return fail(TwGoldenStatus::MissingField, "race.required_*");
    if (rc.words[0] != ctx.requiredCps || rl.words[0] != ctx.requiredLaps) return fail(TwGoldenStatus::MapConstantMismatch, "race.required_*");
    return {};
}

// TwinState -> golden words (every field ImportGolden reads, plus the non-state ones)
template <class Put>
inline void ExportGolden(const TwinState &s, const TwGoldenContext &ctx, Put &&put) {
    std::size_t n = 0;
    const TwFieldInfo *fields = TwinFields(&n);
    std::uint32_t w[16];
    for (std::size_t i = 0; i < n; ++i) {
        const TwFieldInfo &f = fields[i];
        switch (f.codec) {
        case TwCodec::DIRECT:
        case TwCodec::BOOL:
        case TwCodec::NARROW:
            for (int k = 0; k < f.count; ++k) w[k] = detail::LoadElement(s, f, k);
            put(f.golden, w, f.count);
            break;
        case TwCodec::PEERZ:
            for (int k = 0; k < f.count; ++k) { w[3 * k] = 0u; w[3 * k + 1] = 0u; w[3 * k + 2] = detail::LoadElement(s, f, k); }
            put(f.golden, w, 3u * f.count);
            break;
        case TwCodec::SIGN: {
            const std::int8_t v = static_cast<std::int8_t>(detail::LoadElement(s, f, 0));
            w[0] = v == 0 ? 0u : (v > 0 ? 0x3f800000u : 0xbf800000u);
            put(f.golden, w, 1u);
            break;
        }
        case TwCodec::FSLDS: {
            const bool g = detail::LoadElement(s, f, 0) != 0u;
            w[0] = g ? ctx.fsGroundedBits : ctx.fsAirborneBits;
            put("dyna.force_scale", w, 1u);
            w[0] = g ? ctx.ldsGroundedBits : ctx.ldsAirborneBits;
            put("dyna.linear_damping", w, 1u);
            break;
        }
        case TwCodec::SLOTS:
            for (int k = 0; k < 8; ++k) w[k] = k < f.count ? detail::LoadElement(s, f, k) : 0u;
            put(f.golden, w, 8u);
            break;
        case TwCodec::FINLO: {
            const bool present = s.finishPresent != 0u;
            const std::uint64_t lo = s.finishLowerNs[0] | (static_cast<std::uint64_t>(s.finishLowerNs[1]) << 32);
            const std::uint64_t hi = present ? lo + 1u : 0u;
            w[0] = present ? 1u : 0u;
            w[1] = static_cast<std::uint32_t>(lo); w[2] = static_cast<std::uint32_t>(lo >> 32);
            w[3] = static_cast<std::uint32_t>(hi); w[4] = static_cast<std::uint32_t>(hi >> 32);
            w[5] = w[3]; w[6] = w[4];
            put(f.golden, w, 7u);
            break;
        }
        case TwCodec::SPAWN: {
            const std::uint32_t idx = detail::LoadElement(s, f, 0);
            if (idx < ctx.spawns.count) put(f.golden, ctx.spawns.iso[idx], 12u);
            break;
        }
        case TwCodec::FINPR:
        case TwCodec::NONE:
            break;
        }
    }
    // non-state fields
    std::uint32_t z[3] = {0u, 0u, 0u};
    put("ct.wheel_contacts", z, 1u);
    put("ct.body_contacts", z, 1u);
    put("ct.body_normal_sum", z, 3u);
    w[0] = ctx.requiredCps; put("race.required_cps", w, 1u);
    w[0] = ctx.requiredLaps; put("race.required_laps", w, 1u);
    w[0] = ctx.ctlGasBits; put("ctl.gas", w, 1u);
    w[0] = ctx.ctlBrakeBits; put("ctl.brake", w, 1u);
    w[0] = ctx.ctlSteerBits; put("ctl.steer", w, 1u);
}

}  // namespace model
}  // namespace tw

#endif
