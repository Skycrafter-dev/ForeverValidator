// Twin FV adapter: FV runtime clone -> TwinState (import.h).
#include "import.h"

#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "src/twin/common/twin_fv_state.h"
#include "twin_api_internal.h"
#include "twin_world_packs.h"

namespace tw::fv {

namespace {

// harness state words: name -> (offset, count), in StateFields() order (A's WordIndex)
const std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> &WordIndex() {
    static const std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> index = [] {
        std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> at;
        std::uint32_t off = 0u;
        for (const auto &f : twin_fv::StateFields()) {
            at[f.name] = {off, f.count};
            off += f.count;
        }
        return at;
    }();
    return index;
}

}  // namespace

bool ImportFvClone(const ReplaySimulationInstanceClone &clone, const tw::model::TwGoldenContext &context,
                   std::uint32_t seed, tw::model::TwinState *out, std::string *diag) {
    if (out == nullptr) return false;
    std::vector<std::uint32_t> words(twin_fv::StateWordCount());
    twin_fv::ExtractState(clone, words.data());
    const auto &index = WordIndex();
    const tw::model::TwGoldenError e = tw::model::ImportGolden(
            [&](const char *name) -> tw::model::TwGoldenWords {
                const auto it = index.find(name);
                if (it == index.end()) return {};
                return {words.data() + it->second.first, it->second.second};
            },
            context, out);
    if (e.status != tw::model::TwGoldenStatus::Ok) {
        if (diag)
            *diag = std::string("twin import: ") + tw::model::TwGoldenStatusName(e.status) + " (" + e.field + ")";
        return false;
    }
    out->seed = seed;
    return true;
}

bool ImportFvClone(const forevervalidator::experimental::twin::TwinWorld &world,
                   const ReplaySimulationInstanceClone &clone, std::uint32_t seed, tw::model::TwinState *out,
                   std::string *diag) {
    const auto &packs = world.Internal().packs;
    if (!packs) {
        if (diag) *diag = "twin import: the world has no compiled packs";
        return false;
    }
    return ImportFvClone(clone, packs->golden, seed, out, diag);
}

}  // namespace tw::fv
