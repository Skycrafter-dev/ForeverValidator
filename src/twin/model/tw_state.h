// Twin per-candidate state, generated from state_fields.inc (design.md §3).
//
//   TwinState         AoS, 448 B, 64-byte aligned, no padding holes, reserved
//                     bytes zero; trivially copyable: a snapshot is a memcpy.
//   kTwinFields       field table (name, golden name, codec, type, count,
//                     byte offset, life) for import/export, hashing, dumps.
//   TwinStateHash     FNV-1a 64 over the 448 bytes.
//   TwinLanes<L>      SoA batch form: one lane word per logical element
//                     (u8/u16/i8 fields widen to 32-bit lane words).
//   TwinLaneView<L>   accessors for one lane of a TwinLanes<L> group.
//
// FV-free, standard C++17, header-only.
#ifndef TW_STATE_H
#define TW_STATE_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace tw {
namespace model {

enum class TwType : std::uint8_t { F32, U32, U16, U8, I8 };
enum class TwCodec : std::uint8_t { DIRECT, BOOL, NARROW, PEERZ, SIGN, FSLDS, SLOTS, FINLO, FINPR, SPAWN, NONE };
enum class TwLife : std::uint8_t { P, S, T };

namespace detail {
template <TwType T> struct CType;
template <> struct CType<TwType::F32> { using type = float; };
template <> struct CType<TwType::U32> { using type = std::uint32_t; };
template <> struct CType<TwType::U16> { using type = std::uint16_t; };
template <> struct CType<TwType::U8> { using type = std::uint8_t; };
template <> struct CType<TwType::I8> { using type = std::int8_t; };
// Lane word type of the SoA form: floats stay floats, integers widen to 32 bits.
template <TwType T> struct LaneType { using type = std::uint32_t; };
template <> struct LaneType<TwType::F32> { using type = float; };
template <> struct LaneType<TwType::I8> { using type = std::int32_t; };
}  // namespace detail

#define TW_CT(type) typename ::tw::model::detail::CType<::tw::model::TwType::type>::type

// ---- AoS ---------------------------------------------------------------------
namespace detail {
struct TwinStateFields {
#define TW_S(name, ty, golden, codec, life) detail::CType<TwType::ty>::type name;
#define TW_A(name, ty, count, golden, codec, life) detail::CType<TwType::ty>::type name[count];
#include "state_fields.inc"
#undef TW_S
#undef TW_A
};
}  // namespace detail

constexpr std::size_t kTwinStateBytes = 448u;

struct alignas(64) TwinState : detail::TwinStateFields {
    std::uint8_t reserved[kTwinStateBytes - sizeof(detail::TwinStateFields)];
};

static_assert(sizeof(TwinState) == kTwinStateBytes, "TwinState must be 448 bytes");
static_assert(alignof(TwinState) == 64, "TwinState must be 64-byte aligned");
static_assert(std::is_trivially_copyable<TwinState>::value, "TwinState must be trivially copyable");
static_assert(std::is_standard_layout<detail::TwinStateFields>::value, "TwinState fields must be standard layout");

// Field sizes summed in list order: equal to the struct size iff there is no
// padding between fields (the list is ordered by size class).
namespace detail {
constexpr std::size_t kFieldBytesSum = 0u
#define TW_S(name, ty, golden, codec, life) + sizeof(detail::CType<TwType::ty>::type)
#define TW_A(name, ty, count, golden, codec, life) + (count) * sizeof(detail::CType<TwType::ty>::type)
#include "state_fields.inc"
#undef TW_S
#undef TW_A
        ;
}  // namespace detail
static_assert(detail::kFieldBytesSum == sizeof(detail::TwinStateFields), "padding hole in TwinState: order fields by size class");

// ---- field table ---------------------------------------------------------------
struct TwFieldInfo {
    const char *name;
    const char *golden;
    TwCodec codec;
    TwType type;
    std::uint16_t count;
    std::uint16_t offset;      // byte offset in TwinState
    std::uint16_t elementBytes;
    TwLife life;
};

namespace detail {
template <class T, std::size_t N> constexpr std::uint16_t CountOf(T (&)[N]) { return static_cast<std::uint16_t>(N); }
}

inline const TwFieldInfo *TwinFields(std::size_t *count) {
    static const TwFieldInfo table[] = {
#define TW_S(name, ty, golden, codec, life) \
    {#name, golden, TwCodec::codec, TwType::ty, 1u, static_cast<std::uint16_t>(offsetof(detail::TwinStateFields, name)), \
     static_cast<std::uint16_t>(sizeof(detail::CType<TwType::ty>::type)), TwLife::life},
#define TW_A(name, ty, count, golden, codec, life) \
    {#name, golden, TwCodec::codec, TwType::ty, static_cast<std::uint16_t>(count), \
     static_cast<std::uint16_t>(offsetof(detail::TwinStateFields, name)), \
     static_cast<std::uint16_t>(sizeof(detail::CType<TwType::ty>::type)), TwLife::life},
#include "state_fields.inc"
#undef TW_S
#undef TW_A
    };
    *count = sizeof(table) / sizeof(table[0]);
    return table;
}

// ---- hash ----------------------------------------------------------------------
inline std::uint64_t TwinStateHash(const TwinState &s) {
    const unsigned char *p = reinterpret_cast<const unsigned char *>(&s);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < sizeof(TwinState); ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

inline void TwinStateClear(TwinState *s) { std::memset(s, 0, sizeof(TwinState)); }

// ---- SoA lanes -------------------------------------------------------------------
template <int L>
struct alignas(64) TwinLanes {
    static_assert(L == 1 || L == 4 || L == 8 || L == 16, "lane width");
#define TW_S(name, ty, golden, codec, life) alignas(64) typename detail::LaneType<TwType::ty>::type name[1][L];
#define TW_A(name, ty, count, golden, codec, life) alignas(64) typename detail::LaneType<TwType::ty>::type name[count][L];
#include "state_fields.inc"
#undef TW_S
#undef TW_A
};

namespace detail {
template <class Dst, class Src> inline Dst LaneCast(Src v) { return static_cast<Dst>(v); }
}

// Unpack one AoS state into lane `lane` (refill) and pack it back (result, snapshot).
template <int L>
inline void TwinLoadLane(TwinLanes<L> *g, int lane, const TwinState &s) {
#define TW_S(name, ty, golden, codec, life) \
    g->name[0][lane] = detail::LaneCast<typename detail::LaneType<TwType::ty>::type>(s.name);
#define TW_A(name, ty, count, golden, codec, life) \
    for (int k = 0; k < (count); ++k) g->name[k][lane] = detail::LaneCast<typename detail::LaneType<TwType::ty>::type>(s.name[k]);
#include "state_fields.inc"
#undef TW_S
#undef TW_A
}

template <int L>
inline void TwinStoreLane(const TwinLanes<L> &g, int lane, TwinState *s) {
    TwinStateClear(s);
#define TW_S(name, ty, golden, codec, life) s->name = detail::LaneCast<detail::CType<TwType::ty>::type>(g.name[0][lane]);
#define TW_A(name, ty, count, golden, codec, life) \
    for (int k = 0; k < (count); ++k) s->name[k] = detail::LaneCast<detail::CType<TwType::ty>::type>(g.name[k][lane]);
#include "state_fields.inc"
#undef TW_S
#undef TW_A
}

// Read access to one lane, with the AoS types (evaluators, debugging).
template <int L>
struct TwinLaneView {
    const TwinLanes<L> *g;
    int lane;
#define TW_S(name, ty, golden, codec, life) \
    detail::CType<TwType::ty>::type name() const { return detail::LaneCast<detail::CType<TwType::ty>::type>(g->name[0][lane]); }
#define TW_A(name, ty, count, golden, codec, life) \
    detail::CType<TwType::ty>::type name(int k) const { return detail::LaneCast<detail::CType<TwType::ty>::type>(g->name[k][lane]); }
#include "state_fields.inc"
#undef TW_S
#undef TW_A
};

#undef TW_CT

}  // namespace model
}  // namespace tw

#endif
