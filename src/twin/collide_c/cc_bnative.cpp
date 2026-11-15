// Architect C: B's stage (src/twin/collide) built into the engine bench with the bench's own flags,
// under renamed namespaces (compiled with -Dcollide=bnat_collide -Dcq=bnat_cq -Dexact=bnat_exact
// -Dcollide_c=bnat_collide_c, as are B's sources and a second copy of C's in this target) so it links
// beside the FV twin library's copy without clashing. Lets cc_enginebench compare B and C at the same
// ISA in the real engine, and B with C's noise-level narrow phase as its hook (noiseNarrow).
#include <cstring>
#include <memory>
#include <string>

#include "cc_stage.h"
#include "tw_collide.h"

namespace tw::ccbench {

std::unique_ptr<step::CollisionStage> MakeBStage(const world::WorldPackView &view, const void *carShapes, std::size_t carBytes, std::string *err,
                                                 bool noiseNarrow) {
    collide::CarShapes car;
    if (carBytes != sizeof car) { if (err) *err = "CarShapes layout differs"; return nullptr; }
    std::memcpy(&car, carShapes, sizeof car);
    auto s = std::make_unique<collide::ExactCollision>();
    collide::CollideOptions opt;
    if (noiseNarrow) opt.narrow = &collide_c::noise::Narrow;
    if (!s->Init(view, car, err, opt)) return nullptr;
    return s;
}

}  // namespace tw::ccbench
