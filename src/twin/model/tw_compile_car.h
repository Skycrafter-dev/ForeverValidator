// CarPack compile from src/twin/model/stadiumcar-values.json (FV-free).
#ifndef TW_COMPILE_CAR_H
#define TW_COMPILE_CAR_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tw_car_pack.h"

namespace tw {
namespace model {

// Parses the JSON file and fills `pack`. Returns false with `error` set on any
// missing or malformed value, or if the car is not the structure the twin
// specialises on (handling model 5, wheel force mode 2, 4 wheels, 8 ellipsoids).
bool CompileCarFromJson(const std::string &path, TwCarPack *pack, std::string *error);

// Every JSON value the pack consumed, re-exported from the pack: path and the
// 32-bit pattern (f32 bits, u32 value, bool 0/1) or, for the texture, its hex
// string. Used by the bit-equality test against the JSON.
struct TwPackValue {
    std::string path;
    std::uint32_t word = 0u;
    std::string text;   // non-empty for string values (texture hex)
};
std::vector<TwPackValue> ExportCarPackValues(const TwCarPack &pack);

// The JSON's own values, same form (tests: pack == JSON bit for bit, coverage).
struct TwJsonValue {
    std::string type;
    std::uint32_t word = 0u;
    std::string text;
};
bool ReadCarJsonValues(const std::string &path, std::vector<std::pair<std::string, TwJsonValue>> *out, std::string *error);

}  // namespace model
}  // namespace tw

#endif
