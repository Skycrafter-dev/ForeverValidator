#include "tw_world.h"

#include <cstring>

namespace tw {

namespace {
// TWREC003: FvContact carries the static actor. TWREC002 files still load (actor = kNoActor).
constexpr char kMagic[8] = {'T', 'W', 'R', 'E', 'C', '0', '0', '3'};
constexpr char kMagicV2[8] = {'T', 'W', 'R', 'E', 'C', '0', '0', '2'};
struct FvContactV2 {
    std::uint8_t shape, primary, material, pad;
    float P[3], N[3], Sep[3], K[3];
};
static_assert(sizeof(FvContactV2) == 52, "TWREC002 contact layout");
static_assert(sizeof(FvContact) == 56, "TWREC003 contact layout");

template <class T>
bool WriteVec(FILE *f, const std::vector<T> &v) {
    const std::uint64_t n = v.size();
    return std::fwrite(&n, sizeof n, 1, f) == 1 && (n == 0 || std::fwrite(v.data(), sizeof(T), n, f) == n);
}
template <class T>
bool ReadVec(FILE *f, std::vector<T> *v) {
    std::uint64_t n = 0;
    if (std::fread(&n, sizeof n, 1, f) != 1 || n > (1ull << 32)) return false;
    v->resize(n);
    return n == 0 || std::fread(v->data(), sizeof(T), n, f) == n;
}
bool ReadContacts(FILE *f, bool v2, std::vector<FvContact> *out) {
    if (!v2) return ReadVec(f, out);
    std::vector<FvContactV2> old;
    if (!ReadVec(f, &old)) return false;
    out->resize(old.size());
    for (std::size_t i = 0; i < old.size(); ++i) {
        FvContact &c = (*out)[i];
        std::memcpy(&c, &old[i], sizeof(FvContactV2));
        c.actor = kNoActor;
    }
    return true;
}
}  // namespace

bool SaveRecording(const char *path, const Recording &rec) {
    FILE *f = std::fopen(path, "wb");
    if (!f) return false;
    bool ok = std::fwrite(kMagic, 1, 8, f) == 8;
    const std::uint32_t hdr[6] = {static_cast<std::uint32_t>(rec.name.size()), rec.ticks, rec.world.actorCount,
                                  rec.world.surfaceCount, rec.world.triggerTris, rec.world.skippedSurfaces};
    ok = ok && std::fwrite(hdr, sizeof hdr, 1, f) == 1;
    ok = ok && std::fwrite(rec.name.data(), 1, rec.name.size(), f) == rec.name.size();
    ok = ok && std::fwrite(&rec.world.bounds, sizeof(Box3), 1, f) == 1;
    ok = ok && WriteVec(f, rec.world.tris) && WriteVec(f, rec.world.extra) && WriteVec(f, rec.substeps) && WriteVec(f, rec.contacts);
    ok = ok && WriteVec(f, rec.merged) && WriteVec(f, rec.source.surfaces) && WriteVec(f, rec.source.vertices) && WriteVec(f, rec.source.tris);
    ok = (std::fclose(f) == 0) && ok;
    return ok;
}

bool LoadRecording(const char *path, Recording *rec) {
    FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[8];
    std::uint32_t hdr[6];
    bool ok = std::fread(magic, 1, 8, f) == 8;
    const bool v2 = ok && std::memcmp(magic, kMagicV2, 8) == 0;
    ok = ok && (v2 || std::memcmp(magic, kMagic, 8) == 0) && std::fread(hdr, sizeof hdr, 1, f) == 1;
    if (ok) {
        rec->name.resize(hdr[0]);
        ok = std::fread(&rec->name[0], 1, hdr[0], f) == hdr[0];
        rec->ticks = hdr[1]; rec->world.actorCount = hdr[2]; rec->world.surfaceCount = hdr[3];
        rec->world.triggerTris = hdr[4]; rec->world.skippedSurfaces = hdr[5];
    }
    ok = ok && std::fread(&rec->world.bounds, sizeof(Box3), 1, f) == 1;
    ok = ok && ReadVec(f, &rec->world.tris) && ReadVec(f, &rec->world.extra) && ReadVec(f, &rec->substeps) && ReadContacts(f, v2, &rec->contacts);
    ok = ok && ReadContacts(f, v2, &rec->merged) && ReadVec(f, &rec->source.surfaces) && ReadVec(f, &rec->source.vertices) && ReadVec(f, &rec->source.tris);
    std::fclose(f);
    return ok;
}

}  // namespace tw
