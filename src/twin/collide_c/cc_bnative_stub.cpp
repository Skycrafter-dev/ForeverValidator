// Architect C: stand-in for cc_bnative.cpp when B's sources are not built into the engine bench
// (CC_BNAT=OFF): no B stage (use CC_BASE or CC_BLIB).
#include <cstddef>
#include <memory>
#include <string>

#include "stages.h"
#include "tw_world_pack.h"

namespace tw::ccbench {

std::unique_ptr<step::CollisionStage> MakeBStage(const world::WorldPackView &, const void *, std::size_t, std::string *err, bool) {
    if (err) *err = "B's stage not built into this bench (CC_BNAT=OFF)";
    return nullptr;
}

}  // namespace tw::ccbench
