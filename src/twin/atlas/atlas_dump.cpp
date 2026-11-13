// atlas_dump: per real collision call (a *.cposes stream: 11 floats per call, pos, quat xyzw, absorb x4, with the
// world's CarShapes), the shapes the atlas sends to the fallback, for D's S2 timing (tw_atlas_fb_check).
//
//   atlas_dump PACK CARSHAPES CPOSES OUT
//
// OUT, per call: u8 mask (bit k: shape k goes to the fallback), then per set bit in shape order: u32 n and n u32
// world triangle ids, the triangles whose AABB overlaps the shape's world AABB inflated by 5 cm, deduplicated and
// sorted by TriInfo::fvOrder.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "atlas_stage.h"

using namespace tw;

int main(int argc, char **argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: atlas_dump PACK CARSHAPES CPOSES OUT\n"); return 2; }
    world::WorldPack pack;
    std::string err;
    if (!world::LoadWorldPack(argv[1], &pack, &err)) { std::fprintf(stderr, "pack: %s\n", err.c_str()); return 1; }
    world::WorldPackView view;
    if (!world::OpenWorldPack(pack, &view, &err, true)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    collide::CarShapes car;
    {
        FILE *f = std::fopen(argv[2], "rb");
        if (!f || std::fread(&car, sizeof car, 1, f) != 1) { std::fprintf(stderr, "carshapes: cannot read %s\n", argv[2]); return 1; }
        std::fclose(f);
        if (car.magic != collide::CarShapes::kMagic) { std::fprintf(stderr, "carshapes: bad magic\n"); return 1; }
    }
    std::vector<float> calls;
    {
        FILE *f = std::fopen(argv[3], "rb");
        if (!f) { std::fprintf(stderr, "cposes: cannot read %s\n", argv[3]); return 1; }
        float x[11];
        while (std::fread(x, sizeof x, 1, f) == 1) calls.insert(calls.end(), x, x + 11);
        std::fclose(f);
    }
    atlas::AtlasCollision st;
    if (!st.Init(view, car, &err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    FILE *out = std::fopen(argv[4], "wb");
    if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
    atlas::AtlasCache cache;
    const std::size_t N = calls.size() / 11;
    std::size_t shapesFb = 0, callsFb = 0, cands = 0;
    std::vector<std::uint32_t> list;
    for (std::size_t i = 0; i < N; ++i) {
        const float *x = &calls[11 * i];
        collide::ShapePose sp[8];
        collide::ShapePoses(car, x, x + 3, x + 7, sp);
        std::uint8_t mask = 0;
        for (int k = 0; k < 8; ++k) {
            model::RespContact c;
            atlas::FallbackWhy why;
            if (st.Decide(sp[k], k, &c, cache, nullptr, &why) == atlas::kFallback) mask = static_cast<std::uint8_t>(mask | (1u << k));
        }
        std::fwrite(&mask, 1, 1, out);
        if (mask) ++callsFb;
        for (int k = 0; k < 8; ++k) {
            if (!(mask & (1u << k))) continue;
            ++shapesFb;
            Box3 b;
            for (int a = 0; a < 3; ++a) {
                float h = 0.0f;
                for (int j = 0; j < 3; ++j) h += std::fabs(sp[k].world.r[3 * j + a]) * sp[k].radii[j];
                b.lo[a] = sp[k].world.t[a] - h - 0.05f; b.hi[a] = sp[k].world.t[a] + h + 0.05f;
            }
            list.clear();
            st.Exact().Bvh().Query(b, [&](std::uint32_t t) { list.push_back(t); });
            std::sort(list.begin(), list.end(), [&](std::uint32_t a, std::uint32_t c) { return view.triInfo[a].fvOrder < view.triInfo[c].fvOrder; });
            list.erase(std::unique(list.begin(), list.end()), list.end());
            const std::uint32_t n = static_cast<std::uint32_t>(list.size());
            cands += n;
            std::fwrite(&n, 4, 1, out);
            if (n) std::fwrite(list.data(), 4, n, out);
        }
    }
    std::fclose(out);
    std::printf("%zu calls, %zu with a fallback shape (%.1f %%), %.3f fallback shapes per call, %.1f candidates per fallback shape\n", N, callsFb,
                100.0 * callsFb / std::max<std::size_t>(1, N), double(shapesFb) / std::max<std::size_t>(1, N),
                double(cands) / std::max<std::size_t>(1, shapesFb));
    return 0;
}
