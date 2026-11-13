// M1 acceptance test (FV-free): car pack vs JSON, curve unit checks, golden round trip,
// lane round trip. Usage:
//   tw_m1_test --json src/twin/model/stadiumcar-values.json [--golden gate1.tgold ...]
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../tw_compile_car.h"
#include "../tw_state.h"
#include "../tw_state_golden.h"
#include "src/twin/common/twin_golden.h"

using namespace tw::model;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static void TestPack(const std::string &json, TwCarPack *pack) {
    std::string err;
    const bool ok = CompileCarFromJson(json, pack, &err);
    CHECK(ok, "compile: %s", err.c_str());
    if (!ok) return;
    std::vector<std::pair<std::string, TwJsonValue>> js;
    CHECK(ReadCarJsonValues(json, &js, &err), "read json: %s", err.c_str());
    std::map<std::string, TwJsonValue> jm(js.begin(), js.end());
    const std::vector<TwPackValue> pv = ExportCarPackValues(*pack);
    std::set<std::string> exported;
    int mismatch = 0;
    for (const TwPackValue &v : pv) {
        exported.insert(v.path);
        auto it = jm.find(v.path);
        if (it == jm.end()) { ++mismatch; std::printf("  pack path not in json: %s\n", v.path.c_str()); continue; }
        const bool same = v.text.empty() ? (it->second.word == v.word && it->second.text.empty()) : (it->second.text == v.text);
        if (!same) { ++mismatch; std::printf("  mismatch %s: pack 0x%08x json 0x%08x\n", v.path.c_str(), v.word, it->second.word); }
    }
    // coverage: JSON values (StadiumCar, not FV defaults) the pack does not carry
    std::vector<std::string> notCarried;
    for (const auto &kv : js)
        if (kv.first.compare(0, 8, "default.") != 0 && exported.count(kv.first) == 0u) notCarried.push_back(kv.first);
    std::map<std::string, int> groups;
    for (const std::string &p : notCarried) groups[p.substr(0, p.find(']') == std::string::npos ? p.find('.', p.find('.') + 1) : p.find(']') + 1)]++;
    std::printf("pack: %zu values exported, %d mismatches vs JSON; %zu JSON values not carried:", pv.size(), mismatch, notCarried.size());
    for (const auto &g : groups) std::printf(" %s(%d)", g.first.c_str(), g.second);
    std::printf("\n");
    CHECK(mismatch == 0, "pack differs from JSON");
    // the values not carried are duplicates of derived.wheelSet (or an RGB hash): prove it
    int dupMismatch = 0;
    auto jw = [&](const std::string &p) { auto it = jm.find(p); return it == jm.end() ? 0xdeadbeefu : it->second.word; };
    const char *xyz[3] = {"x", "y", "z"};
    for (int w = 0; w < 4; ++w) {
        const std::string sw = "solid.wheels[" + std::to_string(w) + "].", dw = "derived.wheelSet.wheels[" + std::to_string(w) + "].";
        const std::string cs = "vehicle.wheelContactSettings[" + std::to_string(w) + "].";
        for (int k = 0; k < 3; ++k) {
            dupMismatch += jw(sw + "forceApplicationPoint." + xyz[k]) != jw(dw + "forceApplicationPoint." + xyz[k]);
            dupMismatch += jw(sw + "restSurfacePose.translation." + xyz[k]) != jw(dw + "restSurfacePose.translation." + xyz[k]);
            for (int b = 0; b < 3; ++b) {
                const char *basis[3] = {"basisX", "basisY", "basisZ"};
                const std::uint32_t want = b == k ? 0x3f800000u : 0u;   // rest rotation is identity
                dupMismatch += jw(sw + "restSurfacePose.rotation." + basis[b] + "." + xyz[k]) != want;
            }
        }
        dupMismatch += jw(sw + "rollingRadius") != jw(dw + "rollingRadius");
        dupMismatch += jw(sw + "surfaceId") != jw(dw + "surfaceId");
        dupMismatch += jw(cs + "surfaceId") != jw(dw + "surfaceId");
        dupMismatch += jw(cs + "killsLateralSpeedOnContact") != jw(dw + "killsLateralSpeedOnContact");
        dupMismatch += jw(cs + "axle") != jw(dw + "axle");
    }
    std::printf("uncarried duplicates (solid.wheels, wheelContactSettings) vs derived.wheelSet: %d mismatches\n", dupMismatch);
    CHECK(dupMismatch == 0, "uncarried wheel duplicates differ");
    std::printf("folded: invMass %.9g invInertia (%.9g %.9g %.9g) com (%.9g %.9g %.9g) span %.9g wMax %.9g fs %08x/%08x lds %08x/%08x\n",
                pack->invMass, pack->invInertiaDiag[0], pack->invInertiaDiag[1], pack->invInertiaDiag[2], pack->com[0], pack->com[1],
                pack->com[2], pack->wheelLongitudinalSpan, pack->maxAngularSpeed, pack->fsGroundedBits, pack->fsAirborneBits,
                pack->ldsGroundedBits, pack->ldsAirborneBits);
    std::printf("shapes -> wheels:");
    for (int s = 0; s < 8; ++s) std::printf(" %d", pack->shape[s].wheelIndex);
    std::printf("\n");
}

static void TestCurves() {
    TwCurve c;
    const float x[3] = {0.0f, 100.0f, 200.0f}, v[3] = {10.0f, 20.0f, 0.0f};
    CHECK(TwCurveCompile(&c, 3, x, v, true), "compile");
    CHECK(TwCurveEval(c, -5.0f) == 10.0f, "clamp low");
    CHECK(TwCurveEval(c, 500.0f) == 0.0f, "clamp high");
    CHECK(TwCurveEval(c, 50.0f) == 15.0f, "linear mid %g", TwCurveEval(c, 50.0f));
    CHECK(TwCurveEval(c, 150.0f) == 10.0f, "linear mid2");
    CHECK(TwCurveEvalSpeed(c, 100.0f / 3.6f) == TwCurveEval(c, (100.0f / 3.6f) * 3.6f), "speed form");
    TwCurve k;
    CHECK(TwCurveCompile(&k, 3, x, v, false), "compile const");
    CHECK(TwCurveEval(k, 150.0f) == 20.0f, "constant = lower key");
    TwCurve e;
    CHECK(TwCurveCompile(&e, 0, x, v, true) && TwCurveEval(e, 1.0f) == 0.0f, "empty curve = 0");
}

struct GoldenRecordView {
    const twin::File *file;
    const std::uint32_t *rec;
    std::string prefix;
    TwGoldenWords operator()(const char *name) const {
        const twin::FieldSpec *f = file->table.Find(prefix + name);
        if (f == nullptr) return {};
        return {rec + f->offset, f->count};
    }
};

static void TestGolden(const std::string &path, const TwCarPack &pack) {
    twin::File file;
    std::string err;
    if (!twin::ReadFile(path, &file, &err)) { CHECK(false, "read %s: %s", path.c_str(), err.c_str()); return; }
    const twin::FieldSpec *fReplay = file.table.Find("meta.replay");
    // per replay: spawn table (distinct isos seen) and map constants
    std::map<std::uint32_t, std::vector<std::array<std::uint32_t, 12>>> spawns;
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> consts;
    for (std::size_t r = 0; r < file.RecordCount(); ++r) {
        const std::uint32_t *rec = file.Record(r);
        const std::uint32_t rep = fReplay ? rec[fReplay->offset] : 0u;
        for (const char *pre : {"pre.", "post."}) {
            for (const char *fld : {"race.spawn", "race.prev_spawn"}) {
                const twin::FieldSpec *f = file.table.Find(std::string(pre) + fld);
                if (!f) continue;
                std::array<std::uint32_t, 12> iso;
                std::memcpy(iso.data(), rec + f->offset, 48);
                auto &v = spawns[rep];
                bool found = false;
                for (const auto &s : v) found = found || s == iso;
                if (!found) v.push_back(iso);
            }
            const twin::FieldSpec *rc = file.table.Find(std::string(pre) + "race.required_cps");
            const twin::FieldSpec *rl = file.table.Find(std::string(pre) + "race.required_laps");
            if (rc && rl && !consts.count(rep)) consts[rep] = {rec[rc->offset], rec[rl->offset]};
        }
    }
    std::map<std::string, std::size_t> fieldMismatch, fieldChecked;
    std::map<std::string, std::size_t> importErrors;
    std::size_t states = 0, ok = 0, laneMismatch = 0, ctlEchoChecked = 0, ctlEchoMismatch = 0;
    for (std::size_t r = 0; r < file.RecordCount(); ++r) {
        const std::uint32_t *rec = file.Record(r);
        const std::uint32_t rep = fReplay ? rec[fReplay->offset] : 0u;
        const auto &sp = spawns[rep];
        std::vector<std::uint32_t> isoFlat(sp.size() * 12u);
        for (std::size_t i = 0; i < sp.size(); ++i) std::memcpy(&isoFlat[12 * i], sp[i].data(), 48);
        for (const char *pre : {"pre.", "post."}) {
            GoldenRecordView view{&file, rec, pre};
            if (view("body.pos").words == nullptr) continue;
            TwGoldenContext ctx;
            ctx.fsGroundedBits = pack.fsGroundedBits; ctx.ldsGroundedBits = pack.ldsGroundedBits;
            ctx.fsAirborneBits = pack.fsAirborneBits; ctx.ldsAirborneBits = pack.ldsAirborneBits;
            ctx.spawns.iso = reinterpret_cast<const std::uint32_t(*)[12]>(isoFlat.data());
            ctx.spawns.count = static_cast<std::uint32_t>(sp.size());
            ctx.requiredCps = consts[rep].first; ctx.requiredLaps = consts[rep].second;
            // ctl.* is the input echo: the round trip takes it from the record (export context)
            ctx.ctlGasBits = view("ctl.gas").words ? view("ctl.gas").words[0] : 0u;
            ctx.ctlBrakeBits = view("ctl.brake").words ? view("ctl.brake").words[0] : 0u;
            ctx.ctlSteerBits = view("ctl.steer").words ? view("ctl.steer").words[0] : 0u;
            ++states;
            TwinState s;
            const TwGoldenError e = ImportGolden(view, ctx, &s);
            if (e.status != TwGoldenStatus::Ok) { importErrors[std::string(TwGoldenStatusName(e.status)) + ":" + e.field]++; continue; }
            // export and compare word for word
            std::map<std::string, std::vector<std::uint32_t>> outw;
            ExportGolden(s, ctx, [&](const char *name, const std::uint32_t *w, std::uint32_t n) { outw[name].assign(w, w + n); });
            bool all = true;
            for (const auto &kv : outw) {
                const TwGoldenWords g = view(kv.first.c_str());
                fieldChecked[kv.first]++;
                const bool same = g.words != nullptr && g.count == kv.second.size() &&
                                  std::memcmp(g.words, kv.second.data(), 4u * g.count) == 0;
                // w.peer_z: only z is state (FV reads peerAxis.z only); compare z components
                bool sameEff = same;
                if (!same && kv.first == "w.peer_z" && g.words && g.count == 12u) {
                    sameEff = true;
                    for (int w = 0; w < 4; ++w) sameEff = sameEff && g.words[3 * w + 2] == kv.second[3 * w + 2];
                    if (sameEff) { fieldMismatch["w.peer_z(x,y only, not state)"]++; fieldChecked["w.peer_z(x,y only, not state)"]++; }
                }
                if (!sameEff) { fieldMismatch[kv.first]++; all = false; }
            }
            ok += all;
            // lanes: load/store identity at every width
            {
                TwinLanes<16> g16; TwinLanes<4> g4; TwinLanes<1> g1;
                TwinState b16, b4, b1;
                TwinLoadLane(&g16, 7, s); TwinStoreLane(g16, 7, &b16);
                TwinLoadLane(&g4, 3, s); TwinStoreLane(g4, 3, &b4);
                TwinLoadLane(&g1, 0, s); TwinStoreLane(g1, 0, &b1);
                const std::uint64_t h = TwinStateHash(s);
                laneMismatch += (TwinStateHash(b16) != h) + (TwinStateHash(b4) != h) + (TwinStateHash(b1) != h);
                TwinLaneView<16> lv{&g16, 7};
                laneMismatch += lv.rpm() != s.rpm || lv.surface(2) != s.surface[2];
            }
        }
        // input echo claim: post.ctl.* == in.* of the tick
        const twin::FieldSpec *ig = file.table.Find("in.gas"), *pg = file.table.Find("post.ctl.gas");
        const twin::FieldSpec *is = file.table.Find("in.steer"), *ps = file.table.Find("post.ctl.steer");
        if (ig && pg && is && ps) {
            ++ctlEchoChecked;
            ctlEchoMismatch += rec[ig->offset] != rec[pg->offset] || rec[is->offset] != rec[ps->offset];
        }
    }
    std::printf("golden %s: %zu records, %zu states imported+exported, %zu lossless, import errors %zu, lane round-trip mismatches %zu\n",
                path.c_str(), file.RecordCount(), states, ok, [&] { std::size_t n = 0; for (auto &kv : importErrors) n += kv.second; return n; }(),
                laneMismatch);
    for (const auto &kv : importErrors) std::printf("  import error %s: %zu\n", kv.first.c_str(), kv.second);
    for (const auto &kv : fieldMismatch) std::printf("  field %s: %zu of %zu states differ\n", kv.first.c_str(), kv.second, fieldChecked[kv.first]);
    std::printf("  fields round-tripped: %zu; input echo post.ctl == in: %zu of %zu ticks differ\n", fieldChecked.size(), ctlEchoMismatch, ctlEchoChecked);
    std::size_t bad = 0;
    for (const auto &kv : fieldMismatch) if (kv.first.find("not state") == std::string::npos) bad += kv.second;
    CHECK(importErrors.empty(), "import errors");
    CHECK(bad == 0, "lossy fields");
    CHECK(laneMismatch == 0, "lane round trip");
}

int main(int argc, char **argv) {
    std::string json;
    std::vector<std::string> goldens;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
        else if (!std::strcmp(argv[i], "--golden") && i + 1 < argc) goldens.push_back(argv[++i]);
    }
    std::size_t nf = 0;
    const TwFieldInfo *fields = TwinFields(&nf);
    std::size_t used = 0;
    for (std::size_t i = 0; i < nf; ++i) used += fields[i].count * fields[i].elementBytes;
    std::printf("TwinState: %zu bytes (%zu fields, %zu bytes used, %zu reserved)\n", sizeof(TwinState), nf, used, sizeof(TwinState) - used);
    TestCurves();
    static TwCarPack pack;
    if (!json.empty()) TestPack(json, &pack);
    for (const std::string &g : goldens) TestGolden(g, pack);
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
