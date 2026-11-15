// Twin acceptance harness: golden record container (format TWINGOLD v1).
//
// Self-contained (standard library only) so that any engine under test can
// read golden records and write its own outputs without linking FV.
// The byte layout is documented in docs/twin/harness.md ("File format").
//
// A file is a header, a field table and `recordCount` fixed-size records.
// Every field is `count` little-endian 32-bit words of one type (f32, u32 or
// i32). Field names are unique; a record is the concatenation of all fields
// in table order.
#ifndef FOREVERVALIDATOR_TWIN_GOLDEN_H
#define FOREVERVALIDATOR_TWIN_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace twin {

constexpr char kMagic[8] = {'T', 'W', 'I', 'N', 'G', 'O', 'L', 'D'};
constexpr std::uint32_t kFormatVersion = 1u;

enum class FieldType : std::uint8_t { F32 = 0, U32 = 1, I32 = 2 };

inline const char *FieldTypeName(FieldType type) {
    switch (type) {
    case FieldType::F32: return "f32";
    case FieldType::U32: return "u32";
    case FieldType::I32: return "i32";
    }
    return "?";
}

struct FieldSpec {
    std::string name;
    FieldType type = FieldType::F32;
    std::uint16_t count = 1u;
    std::uint32_t offset = 0u;  // word offset inside a record (derived)
};

// File kinds (stored in the header). A golden file holds pre-state, input,
// post-state, events and margins; an engine-output file holds meta.record_id
// plus whatever post.* / ev.* fields the engine produces.
enum class FileKind : std::uint32_t {
    Golden = 1,
    EngineOutput = 2,
};

class Table {
public:
    Table() = default;

    // Appends a field; returns its word offset. Duplicate names are a bug.
    std::uint32_t Add(const std::string &name, FieldType type,
                      std::uint16_t count) {
        FieldSpec spec;
        spec.name = name;
        spec.type = type;
        spec.count = count;
        spec.offset = words_;
        index_[name] = fields_.size();
        fields_.push_back(spec);
        words_ += count;
        return spec.offset;
    }

    const std::vector<FieldSpec> &Fields() const { return fields_; }
    std::uint32_t WordsPerRecord() const { return words_; }

    const FieldSpec *Find(const std::string &name) const {
        auto it = index_.find(name);
        return it == index_.end() ? nullptr : &fields_[it->second];
    }

private:
    std::vector<FieldSpec> fields_;
    std::map<std::string, std::size_t> index_;
    std::uint32_t words_ = 0u;
};

struct File {
    FileKind kind = FileKind::Golden;
    std::string metadata;  // UTF-8 JSON text, informative
    Table table;
    std::vector<std::uint32_t> words;  // recordCount * WordsPerRecord()

    std::size_t RecordCount() const {
        const std::uint32_t w = table.WordsPerRecord();
        return w == 0u ? 0u : words.size() / w;
    }
    const std::uint32_t *Record(std::size_t index) const {
        return words.data() + index * table.WordsPerRecord();
    }
    std::uint32_t *Record(std::size_t index) {
        return words.data() + index * table.WordsPerRecord();
    }
    std::uint32_t *AppendRecord() {
        words.resize(words.size() + table.WordsPerRecord(), 0u);
        return Record(RecordCount() - 1u);
    }
};

inline std::uint32_t FloatBits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float BitsFloat(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Distance in units in the last place between two binary32 values
// (0 for identical bits and for +0/-0; saturates; NaN -> max).
inline std::uint64_t UlpDistance(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    auto ordered = [](float v) -> std::int64_t {
        const std::int32_t bits = static_cast<std::int32_t>(FloatBits(v));
        return bits < 0 ? static_cast<std::int64_t>(INT32_MIN) - bits
                        : static_cast<std::int64_t>(bits);
    };
    const std::int64_t d = ordered(a) - ordered(b);
    return static_cast<std::uint64_t>(d < 0 ? -d : d);
}

namespace detail {

inline void PutU32(std::vector<std::uint8_t> &out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
inline void PutU16(std::vector<std::uint8_t> &out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
}

struct Reader {
    const std::vector<std::uint8_t> &bytes;
    std::size_t at = 0u;
    bool ok = true;
    bool Need(std::size_t n) {
        if (!ok || bytes.size() - at < n) ok = false;
        return ok;
    }
    std::uint32_t U32() {
        if (!Need(4)) return 0u;
        std::uint32_t v = 0u;
        for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(bytes[at + i]) << (8 * i);
        at += 4;
        return v;
    }
    std::uint16_t U16() {
        if (!Need(2)) return 0u;
        const std::uint16_t v = static_cast<std::uint16_t>(
                bytes[at] | (static_cast<std::uint16_t>(bytes[at + 1]) << 8));
        at += 2;
        return v;
    }
    std::uint8_t U8() {
        if (!Need(1)) return 0u;
        return bytes[at++];
    }
    std::string Str(std::size_t n) {
        if (!Need(n)) return {};
        std::string s(reinterpret_cast<const char *>(bytes.data() + at), n);
        at += n;
        return s;
    }
};

}  // namespace detail

// Layout (all integers little-endian):
//   char[8] "TWINGOLD"; u32 version (=1); u32 kind;
//   u32 metadataBytes; metadataBytes of UTF-8 JSON;
//   u32 fieldCount; per field: u16 nameBytes, name, u8 type, u8 0, u16 count;
//   u32 wordsPerRecord; u32 recordCount;
//   recordCount * wordsPerRecord u32 words.
inline bool WriteFile(const std::string &path, const File &file,
                      std::string *error) {
    std::vector<std::uint8_t> out;
    out.insert(out.end(), kMagic, kMagic + 8);
    detail::PutU32(out, kFormatVersion);
    detail::PutU32(out, static_cast<std::uint32_t>(file.kind));
    detail::PutU32(out, static_cast<std::uint32_t>(file.metadata.size()));
    out.insert(out.end(), file.metadata.begin(), file.metadata.end());
    detail::PutU32(out, static_cast<std::uint32_t>(file.table.Fields().size()));
    for (const FieldSpec &f : file.table.Fields()) {
        detail::PutU16(out, static_cast<std::uint16_t>(f.name.size()));
        out.insert(out.end(), f.name.begin(), f.name.end());
        out.push_back(static_cast<std::uint8_t>(f.type));
        out.push_back(0u);
        detail::PutU16(out, f.count);
    }
    detail::PutU32(out, file.table.WordsPerRecord());
    detail::PutU32(out, static_cast<std::uint32_t>(file.RecordCount()));
    const std::size_t header = out.size();
    out.resize(header + file.words.size() * 4u);
    for (std::size_t i = 0; i < file.words.size(); ++i) {
        const std::uint32_t v = file.words[i];
        for (int b = 0; b < 4; ++b) out[header + 4 * i + b] = static_cast<std::uint8_t>(v >> (8 * b));
    }
    std::FILE *fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        if (error) *error = "cannot open " + path + " for writing";
        return false;
    }
    const bool ok = std::fwrite(out.data(), 1, out.size(), fp) == out.size();
    const bool closed = std::fclose(fp) == 0;
    if (!ok || !closed) {
        if (error) *error = "short write to " + path;
        return false;
    }
    return true;
}

inline bool ReadFile(const std::string &path, File *file, std::string *error) {
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::vector<std::uint8_t> bytes;
    std::uint8_t buffer[1 << 16];
    std::size_t n;
    while ((n = std::fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        bytes.insert(bytes.end(), buffer, buffer + n);
    }
    std::fclose(fp);
    detail::Reader r{bytes};
    const std::string magic = r.Str(8);
    if (!r.ok || std::memcmp(magic.data(), kMagic, 8) != 0) {
        if (error) *error = path + ": not a TWINGOLD file";
        return false;
    }
    const std::uint32_t version = r.U32();
    if (version != kFormatVersion) {
        if (error) *error = path + ": unsupported TWINGOLD version " + std::to_string(version);
        return false;
    }
    File result;
    result.kind = static_cast<FileKind>(r.U32());
    result.metadata = r.Str(r.U32());
    const std::uint32_t fieldCount = r.U32();
    for (std::uint32_t i = 0; r.ok && i < fieldCount; ++i) {
        const std::string name = r.Str(r.U16());
        const std::uint8_t type = r.U8();
        r.U8();
        const std::uint16_t count = r.U16();
        if (type > 2u || result.table.Find(name) != nullptr) {
            r.ok = false;
            break;
        }
        result.table.Add(name, static_cast<FieldType>(type), count);
    }
    const std::uint32_t wordsPerRecord = r.U32();
    const std::uint32_t recordCount = r.U32();
    if (!r.ok || wordsPerRecord != result.table.WordsPerRecord() ||
        (bytes.size() - r.at) != static_cast<std::size_t>(recordCount) * wordsPerRecord * 4u) {
        if (error) *error = path + ": corrupt TWINGOLD header or size";
        return false;
    }
    result.words.resize(static_cast<std::size_t>(recordCount) * wordsPerRecord);
    for (std::size_t i = 0; i < result.words.size(); ++i) {
        const std::uint8_t *p = bytes.data() + r.at + 4 * i;
        result.words[i] = static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
                (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
    }
    *file = std::move(result);
    return true;
}

// Minimal JSON string escaping for metadata.
inline std::string JsonEscape(const std::string &s) {
    std::string out;
    for (char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

// Comparison shape of a state field: f32 fields named "*quat" with 4
// values are one quaternion (error = rotation angle, rad); other f32 fields
// with a multiple of 3 values are 3-vectors (error = Euclidean norm);
// everything else is per scalar.
enum class Shape : std::uint8_t { Scalar, Vec3, Quat };

inline Shape ShapeOf(const std::string &stateName, FieldType type, std::uint16_t count) {
    if (type != FieldType::F32) return Shape::Scalar;
    if (count == 4 && stateName.size() >= 4 && stateName.compare(stateName.size() - 4, 4, "quat") == 0) {
        return Shape::Quat;
    }
    if (count % 3 == 0) return Shape::Vec3;
    return Shape::Scalar;
}

inline std::uint32_t GroupWidth(Shape shape) {
    return shape == Shape::Scalar ? 1u : (shape == Shape::Vec3 ? 3u : 4u);
}

// Error and reference magnitude of one group (`width` floats).
inline void GroupError(Shape shape, const float *fv, const float *other, double *error, double *reference) {
    const std::uint32_t width = GroupWidth(shape);
    bool exact = true;
    for (std::uint32_t c = 0; c < width; ++c) exact &= FloatBits(fv[c]) == FloatBits(other[c]);
    if (shape == Shape::Quat) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < 4; ++i) {
            dot += static_cast<double>(fv[i]) * other[i];
            na += static_cast<double>(fv[i]) * fv[i];
            nb += static_cast<double>(other[i]) * other[i];
        }
        const double c = std::fabs(dot) / std::sqrt(std::max(1e-300, na * nb));
        *error = exact ? 0.0 : 2.0 * std::acos(c > 1.0 ? 1.0 : c);
        *reference = 1.0;
    } else {
        double e = 0.0, r = 0.0;
        for (std::uint32_t c = 0; c < width; ++c) {
            const double d = static_cast<double>(other[c]) - fv[c];
            e += d * d;
            r += static_cast<double>(fv[c]) * fv[c];
        }
        *error = std::sqrt(e);
        *reference = std::sqrt(r);
    }
    if (std::isnan(*error)) *error = std::numeric_limits<double>::infinity();
}

// Flat JSON helpers for the generator's metadata: the objects of array
// `arrayKey`, and a string member of one object.
inline std::vector<std::string> JsonObjects(const std::string &text, const std::string &arrayKey) {
    std::vector<std::string> out;
    const std::size_t key = text.find("\"" + arrayKey + "\":[");
    if (key == std::string::npos) return out;
    std::size_t at = text.find('[', key);
    int depth = 0;
    std::size_t start = 0;
    for (std::size_t i = at + 1; i < text.size(); ++i) {
        if (text[i] == '{') {
            if (depth++ == 0) start = i;
        } else if (text[i] == '}') {
            if (--depth == 0) out.push_back(text.substr(start, i - start + 1));
        } else if (text[i] == ']' && depth == 0) {
            break;
        }
    }
    return out;
}

inline std::string JsonString(const std::string &object, const std::string &key) {
    const std::string needle = "\"" + key + "\":\"";
    const std::size_t at = object.find(needle);
    if (at == std::string::npos) return {};
    const std::size_t begin = at + needle.size();
    return object.substr(begin, object.find('"', begin) - begin);
}

// Reason bits stored in meta.reasons (why a tick is interesting).
enum Reason : std::uint32_t {
    kReasonRandom = 1u << 0,         // uniform sample
    kReasonRaceStart = 1u << 1,      // speed-block released this tick
    kReasonAirborne = 1u << 2,       // no wheel contact before and after
    kReasonTakeoff = 1u << 3,        // some wheel contact -> none
    kReasonLanding = 1u << 4,        // none -> some wheel contact
    kReasonWheelChange = 1u << 5,    // any per-wheel contact flag flips
    kReasonBodyContact = 1u << 6,    // body (non-wheel) contact during tick
    kReasonTurbo = 1u << 7,          // turbo active or (re)started
    kReasonTurboStart = 1u << 8,     // turbo type or end tick changed
    kReasonOneSubstep = 1u << 9,     // 1 physics substep
    kReasonManySubsteps = 1u << 10,  // >= 8 substeps
    kReasonSubstepTie = 1u << 11,    // substep metric within 0.02 of a boundary
    kReasonRespawn = 1u << 12,       // respawn executed
    kReasonCheckpoint = 1u << 13,    // checkpoint / finish accepted
    kReasonGearChange = 1u << 14,    // gear index changed
    kReasonSlip = 1u << 15,          // any wheel slipping flag set
    kReasonSurface = 1u << 16,       // a wheel on a non-road material
    kReasonFreewheel = 1u << 17,     // free-wheel flag set
    kReasonBurnout = 1u << 18,       // burnout phase active
    kReasonReverse = 1u << 19,       // reverse flag set
    kReasonGasBrake = 1u << 20,      // gas and brake both held
    kReasonSteerFull = 1u << 21,     // |steer| == 1
    kReasonStraight = 1u << 22,      // 4 wheels on road, no slip, |steer|<.05
    kReasonWater = 1u << 23,         // no-ground-friction guard (water)
    kReasonAirControl = 1u << 24,    // air-control memory refresh flag
    kReasonLateralSlowDown = 1u << 25,
    kReasonFinish = 1u << 26,
    kReasonCount = 27u,
};

inline const char *ReasonName(std::uint32_t bit) {
    static const char *names[kReasonCount] = {
            "random", "race_start", "airborne", "takeoff", "landing",
            "wheel_change", "body_contact", "turbo", "turbo_start",
            "one_substep", "many_substeps", "substep_tie", "respawn",
            "checkpoint", "gear_change", "slip", "surface", "freewheel",
            "burnout", "reverse", "gas_brake", "steer_full", "straight",
            "water", "air_control", "lateral_slowdown", "finish"};
    return bit < kReasonCount ? names[bit] : "?";
}

constexpr std::uint32_t kNoWindow = 0xFFFFFFFFu;

}  // namespace twin

#endif
