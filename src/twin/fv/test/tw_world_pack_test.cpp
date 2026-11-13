// tw_world_pack_test: FV-free checks of the world package (world/tw_world_pack.h).
//
//   tw_world_pack_test [PACK.twpack [REC.twrec]...]
//
// Always: rotation codes (all 48 signed permutations round-trip, general
// rotations refused), hash sensitivity on a synthetic buffer.
// With a pack (written by tw_world_fvcheck --out DIR --keep): it opens and
// validates; a flipped byte, a wrong version, a truncation and a bad
// section size are each refused; the race map is valid; the bench world
// has the pack's triangles.
// With recordings (tw_record, TWREC003): every raw and merged contact's
// actor is a pack actor that owns a triangle of the contact's material, and
// the recording's world equals the pack's bench world on FV's collision set.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "tw_world.h"
#include "tw_world_pack.h"

namespace w = tw::world;

namespace {
int g_fail = 0, g_pass = 0;
void Check(bool ok, const std::string &what) {
    if (ok) {
        ++g_pass;
        return;
    }
    ++g_fail;
    std::printf("FAIL %s\n", what.c_str());
}

void RotationCodes() {
    int roundTrip = 0;
    for (int code = 0; code < 48; ++code) {
        float R[3][3];
        Check(w::RotationFromCode(static_cast<std::uint8_t>(code), R), "RotationFromCode " + std::to_string(code));
        float iso[12] = {};
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) iso[3 * j + i] = R[i][j];
        if (w::RotationCode(iso) == code) ++roundTrip;
    }
    Check(roundTrip == 48, "48 signed permutations round-trip (" + std::to_string(roundTrip) + ")");
    const float yaw45[12] = {0.70710677f, 0, -0.70710677f, 0, 1, 0, 0.70710677f, 0, 0.70710677f, 0, 0, 0};
    Check(w::RotationCode(yaw45) == w::kRotGeneral, "general rotation refused");
    const float twoOnOneRow[12] = {1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0};
    Check(w::RotationCode(twoOnOneRow) == w::kRotGeneral, "singular permutation refused");
}

void HashSensitivity() {
    std::vector<std::uint8_t> b(1000);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::uint8_t>(i * 131u);
    const std::uint64_t h0 = w::PackHash(b.data(), b.size());
    std::set<std::uint64_t> seen{h0};
    for (std::size_t i = 0; i < b.size(); i += 37) {
        b[i] ^= 1u;
        seen.insert(w::PackHash(b.data(), b.size()));
        b[i] ^= 1u;
    }
    Check(seen.size() == 1 + (b.size() + 36) / 37, "single-bit flips change the hash");
    Check(w::PackHash(b.data(), b.size() - 1) != h0, "length changes the hash");
}

bool Copy(const w::WorldPack &src, w::WorldPack *dst, std::size_t size) {
    if (!dst->Allocate(size)) return false;
    std::memcpy(dst->MutableData(), src.data(), std::min(size, src.size()));
    return true;
}

void PackChecks(const char *path, w::WorldPack *pack, w::WorldPackView *view) {
    std::string err;
    Check(w::LoadWorldPack(path, pack, &err), std::string("load ") + path + " " + err);
    Check(w::OpenWorldPack(*pack, view, &err), "open " + err);
    if (!view->header) return;
    w::WorldPack bad;
    w::WorldPackView v2;
    Copy(*pack, &bad, pack->size());
    bad.MutableData()[pack->size() / 2] ^= 0x10u;
    Check(!w::OpenWorldPack(bad, &v2, &err) && err.find("hash") != std::string::npos, "flipped byte refused");
    Copy(*pack, &bad, pack->size());
    reinterpret_cast<w::PackHeader *>(bad.MutableData())->version = w::kPackVersion + 1u;
    Check(!w::OpenWorldPack(bad, &v2, &err), "wrong version refused");
    Copy(*pack, &bad, pack->size() - 64u);
    Check(!w::OpenWorldPack(bad, &v2, &err), "truncation refused");
    Copy(*pack, &bad, pack->size());
    reinterpret_cast<w::PackHeader *>(bad.MutableData())->sections[0].elemBytes = 60u;
    Check(!w::OpenWorldPack(bad, &v2, &err), "bad element size refused");
    Check(tw::race::TwRaceMapValid(view->RaceMap()), "race map valid");
    Check(view->spawns.size() >= 1u && view->triggers.size() == view->triggerVolumes.size(), "race tables");
    tw::World bench;
    w::MakeBenchWorld(*view, &bench);
    Check(bench.tris.size() == view->tris.size() && bench.extra.size() == view->tris.size(), "bench world size");
    std::printf("%s: %zu tris, %zu instances, %zu models, %zu actors, %zu triggers, %zu spawns, hash %016llx, %.2f MB\n", path,
                view->tris.size(), view->instances.size(), view->models.size(), view->actors.size(), view->triggers.size(),
                view->spawns.size(), static_cast<unsigned long long>(pack->Hash()), pack->size() / 1048576.0);
}

void RecordingChecks(const char *path, const w::WorldPackView &v) {
    tw::Recording rec;
    Check(tw::LoadRecording(path, &rec), std::string("load recording ") + path);
    // (actor, material) pairs present in the pack
    std::set<std::uint64_t> owned;
    for (const tw::Tri &t : v.tris) owned.insert((std::uint64_t(t.meta >> tw::kMetaActorShift) << 8) | (t.meta & tw::kMetaMaterialMask));
    std::size_t bad = 0, unknown = 0, triggers = 0, badTrigger = 0;
    auto scan = [&](const std::vector<tw::FvContact> &cs, bool checkMaterial) {
        for (const tw::FvContact &c : cs) {
            if (c.actor == tw::kNoActor || c.actor >= v.actors.size()) {
                ++unknown;
                continue;
            }
            if (checkMaterial && !owned.count((std::uint64_t(c.actor) << 8) | c.material)) ++bad;
            if (v.actors[c.actor].group == 1u) {
                ++triggers;
                if (v.actors[c.actor].trigger == w::kNoTrigger) ++badTrigger;
            }
        }
    };
    // TWREC003 raw contacts carry the mapped material id too (the recorder maps the local index itself).
    scan(rec.contacts, true);
    scan(rec.merged, true);
    Check(unknown == 0, std::string(path) + ": contacts without a pack actor: " + std::to_string(unknown));
    Check(bad == 0, std::string(path) + ": contacts whose actor owns no triangle of that material: " + std::to_string(bad));
    Check(badTrigger == 0, std::string(path) + ": group-1 contacts on non-trigger actors: " + std::to_string(badTrigger));
    tw::World bench;
    w::MakeBenchWorld(v, &bench);
    // the recorder's world is B's prototype compile; on FV's set it must equal the bench world
    std::set<std::uint32_t> surfaces;
    for (const w::Instance &in : v.instances) surfaces.insert(in.fvSurface);
    std::vector<tw::Tri> recOnSet;
    for (std::size_t i = 0; i < rec.world.tris.size(); ++i)
        if (surfaces.count(rec.world.extra[i].surface)) recOnSet.push_back(rec.world.tris[i]);
    Check(recOnSet.size() == bench.tris.size() &&
                  std::memcmp(recOnSet.data(), bench.tris.data(), recOnSet.size() * sizeof(tw::Tri)) == 0,
          std::string(path) + ": recording world == pack bench world on FV's set");
    std::printf("%s: %zu raw + %zu merged contacts, %zu on triggers, all with pack actors\n", path, rec.contacts.size(),
                rec.merged.size(), triggers);
}
}  // namespace

int main(int argc, char **argv) {
    RotationCodes();
    HashSensitivity();
    if (argc > 1) {
        w::WorldPack pack;
        w::WorldPackView view;
        PackChecks(argv[1], &pack, &view);
        for (int i = 2; i < argc && view.header; ++i) RecordingChecks(argv[i], view);
    }
    std::printf("%d passed, %d failed\n%s\n", g_pass, g_fail, g_fail ? "RESULT: FAIL" : "RESULT: PASS");
    return g_fail ? 1 : 0;
}
