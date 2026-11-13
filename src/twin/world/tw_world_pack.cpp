// Twin world package: blob ownership, hash, save/load, validation, views.
// FV-free (docs/twin/m2-world.md).
#include "tw_world_pack.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

namespace tw::world {

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the twin world package is little-endian"
#endif

namespace {

constexpr std::uint32_t kElemBytes[kSectionCount] = {
    sizeof(Tri),          sizeof(TriPlane),       sizeof(TriInfo),         sizeof(Instance),
    sizeof(Model),        sizeof(ModelTri),       sizeof(Vec3f),           sizeof(std::uint32_t),
    sizeof(Actor),        sizeof(race::TwTrigger), sizeof(TriggerVolume), sizeof(race::TwSpawn),
    sizeof(SpecialSurface), sizeof(std::uint8_t)};

template <class T>
PackSpan<T> SpanOf(const WorldPack &pack, Section s) {
    const SectionEntry &e = pack.Header().sections[static_cast<std::uint32_t>(s)];
    PackSpan<T> span;
    span.ptr = e.count ? reinterpret_cast<const T *>(pack.data() + e.offset) : nullptr;
    span.count = static_cast<std::size_t>(e.count);
    return span;
}

bool Fail(std::string *error, const std::string &what) {
    if (error) *error = "world pack: " + what;
    return false;
}

constexpr int kPerm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

}  // namespace

// ---- rotation codes -------------------------------------------------------------------
std::uint8_t RotationCode(const float iso[12]) {
    int row[3];
    int neg[3];
    for (int j = 0; j < 3; ++j) {          // column j = basis j = iso[3j .. 3j+2]
        int found = -1;
        for (int i = 0; i < 3; ++i) {
            const float v = iso[3 * j + i];
            if (v == 1.0f || v == -1.0f) {
                if (found >= 0) return kRotGeneral;
                found = i;
                neg[j] = v < 0.0f ? 1 : 0;
            } else if (v != 0.0f) {
                return kRotGeneral;
            }
        }
        if (found < 0) return kRotGeneral;
        row[j] = found;
    }
    for (int p = 0; p < 6; ++p)
        if (kPerm[p][0] == row[0] && kPerm[p][1] == row[1] && kPerm[p][2] == row[2])
            return static_cast<std::uint8_t>(p * 8 + neg[0] + 2 * neg[1] + 4 * neg[2]);
    return kRotGeneral;  // two columns on one row
}

bool RotationFromCode(std::uint8_t code, float R[3][3]) {
    if (code >= 48u) return false;
    const int p = code / 8;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R[i][j] = 0.0f;
    for (int j = 0; j < 3; ++j) R[kPerm[p][j]][j] = ((code >> j) & 1u) ? -1.0f : 1.0f;
    return true;
}

// ---- blob ------------------------------------------------------------------------------
void WorldPack::Free::operator()(std::uint8_t *p) const {
    ::operator delete[](p, std::align_val_t(kPackAlign));
}

bool WorldPack::Allocate(std::size_t bytes) {
    bytes_.reset();
    size_ = 0u;
    if (bytes < sizeof(PackHeader)) return false;
    std::uint8_t *p = static_cast<std::uint8_t *>(::operator new[](bytes, std::align_val_t(kPackAlign), std::nothrow));
    if (p == nullptr) return false;
    std::memset(p, 0, bytes);
    bytes_.reset(p);
    size_ = bytes;
    return true;
}

std::uint64_t PackHash(const std::uint8_t *data, std::size_t bytes) {
    // Four independent multiply-xorshift lanes over 8-byte words, then a fold.
    const std::uint64_t k0 = 0x9E3779B97F4A7C15ull, k1 = 0xC2B2AE3D27D4EB4Full, k2 = 0x165667B19E3779F9ull,
                        k3 = 0xFF51AFD7ED558CCDull;
    std::uint64_t h[4] = {k0 ^ bytes, k1, k2, k3};
    std::size_t i = 0;
    for (; i + 32 <= bytes; i += 32) {
        std::uint64_t w[4];
        std::memcpy(w, data + i, 32);
        for (int l = 0; l < 4; ++l) {
            h[l] = (h[l] ^ w[l]) * k3;
            h[l] ^= h[l] >> 29;
        }
    }
    std::uint8_t tail[32] = {};
    std::memcpy(tail, data + i, bytes - i);
    for (int l = 0; l < 4; ++l) {
        std::uint64_t w;
        std::memcpy(&w, tail + 8 * l, 8);
        h[l] = (h[l] ^ w) * k3;
        h[l] ^= h[l] >> 29;
    }
    std::uint64_t r = h[0];
    for (int l = 1; l < 4; ++l) {
        r = (r ^ (h[l] + k0 + (r << 6) + (r >> 2))) * k1;
        r ^= r >> 31;
    }
    r ^= r >> 33;
    r *= k3;
    r ^= r >> 33;
    return r;
}

namespace {
std::uint64_t HashWithZeroedField(const std::uint8_t *data, std::size_t bytes) {
    // contentHash lives in the header; hash the header copy with it zeroed, then the rest.
    PackHeader h;
    std::memcpy(&h, data, sizeof h);
    h.contentHash = 0u;
    std::vector<std::uint8_t> head(sizeof h);
    std::memcpy(head.data(), &h, sizeof h);
    const std::uint64_t a = PackHash(head.data(), head.size());
    const std::uint64_t b = PackHash(data + sizeof h, bytes - sizeof h);
    return PackHash(reinterpret_cast<const std::uint8_t *>(&a), 8) ^ (b * 0x9E3779B97F4A7C15ull);
}
}  // namespace

void SealWorldPack(WorldPack *pack) {
    if (pack == nullptr || pack->size() < sizeof(PackHeader)) return;
    PackHeader *h = reinterpret_cast<PackHeader *>(pack->MutableData());
    h->contentHash = 0u;
    h->contentHash = HashWithZeroedField(pack->data(), pack->size());
}

bool SaveWorldPack(const char *path, const WorldPack &pack, std::string *error) {
    if (pack.size() == 0u) return Fail(error, "empty pack");
    FILE *f = std::fopen(path, "wb");
    if (f == nullptr) return Fail(error, std::string("cannot open ") + path);
    const bool ok = std::fwrite(pack.data(), 1, pack.size(), f) == pack.size();
    const bool closed = std::fclose(f) == 0;
    return (ok && closed) || Fail(error, std::string("cannot write ") + path);
}

bool LoadWorldPack(const char *path, WorldPack *pack, std::string *error) {
    FILE *f = std::fopen(path, "rb");
    if (f == nullptr) return Fail(error, std::string("cannot open ") + path);
    PackHeader h;
    bool ok = std::fread(&h, sizeof h, 1, f) == 1 && std::memcmp(h.magic, kPackMagic, 8) == 0 &&
              h.totalBytes >= sizeof h && h.totalBytes < (std::uint64_t(1) << 36);
    ok = ok && pack->Allocate(static_cast<std::size_t>(h.totalBytes));
    if (ok) {
        std::memcpy(pack->MutableData(), &h, sizeof h);
        const std::size_t rest = pack->size() - sizeof h;
        ok = std::fread(pack->MutableData() + sizeof h, 1, rest, f) == rest;
    }
    std::fclose(f);
    if (!ok) return Fail(error, std::string("cannot read ") + path);
    WorldPackView view;
    return OpenWorldPack(*pack, &view, error);
}

// ---- validation ------------------------------------------------------------------------
bool OpenWorldPack(const WorldPack &pack, WorldPackView *view, std::string *error, bool skipHash) {
    if (pack.size() < sizeof(PackHeader)) return Fail(error, "too small");
    const PackHeader &h = pack.Header();
    if (std::memcmp(h.magic, kPackMagic, 8) != 0) return Fail(error, "bad magic");
    if (h.version != kPackVersion) return Fail(error, "version " + std::to_string(h.version) + ", expected " +
                                                          std::to_string(kPackVersion));
    if (h.headerBytes != sizeof(PackHeader) || h.totalBytes != pack.size() || h.sectionCount != kSectionCount)
        return Fail(error, "header sizes");
    for (std::uint32_t s = 0; s < kSectionCount; ++s) {
        const SectionEntry &e = h.sections[s];
        if (e.elemBytes != kElemBytes[s]) return Fail(error, "section " + std::to_string(s) + " element size");
        if (e.offset % kPackAlign != 0u || e.offset < sizeof(PackHeader) || e.offset > pack.size() ||
            e.count > (pack.size() - e.offset) / e.elemBytes)
            return Fail(error, "section " + std::to_string(s) + " bounds");
    }
    if (!skipHash && HashWithZeroedField(pack.data(), pack.size()) != h.contentHash)
        return Fail(error, "content hash mismatch");

    WorldPackView v;
    v.header = &h;
    v.tris = SpanOf<Tri>(pack, Section::Tris);
    v.triPlanes = SpanOf<TriPlane>(pack, Section::TriPlanes);
    v.triInfo = SpanOf<TriInfo>(pack, Section::TriInfo);
    v.instances = SpanOf<Instance>(pack, Section::Instances);
    v.models = SpanOf<Model>(pack, Section::Models);
    v.modelTris = SpanOf<ModelTri>(pack, Section::ModelTris);
    v.modelVerts = SpanOf<Vec3f>(pack, Section::ModelVerts);
    v.modelInstances = SpanOf<std::uint32_t>(pack, Section::ModelInstances);
    v.actors = SpanOf<Actor>(pack, Section::Actors);
    v.triggers = SpanOf<race::TwTrigger>(pack, Section::Triggers);
    v.triggerVolumes = SpanOf<TriggerVolume>(pack, Section::TriggerVolumes);
    v.spawns = SpanOf<race::TwSpawn>(pack, Section::Spawns);
    v.specials = SpanOf<SpecialSurface>(pack, Section::Specials);
    v.waterCells = SpanOf<std::uint8_t>(pack, Section::WaterCells);

    // cross-references
    const std::size_t nt = v.tris.size();
    if (v.triPlanes.size() != nt || v.triInfo.size() != nt) return Fail(error, "triangle side tables");
    if (h.actorCount != v.actors.size()) return Fail(error, "actor count");
    if (v.triggerVolumes.size() != v.triggers.size() || v.triggers.size() >= kNoTrigger) return Fail(error, "trigger tables");
    for (const Model &m : v.models) {
        if (m.firstTri > v.modelTris.size() || m.triCount > v.modelTris.size() - m.firstTri ||
            m.firstVert > v.modelVerts.size() || m.vertCount > v.modelVerts.size() - m.firstVert ||
            m.firstRef > v.modelInstances.size() || m.instanceCount > v.modelInstances.size() - m.firstRef)
            return Fail(error, "model ranges");
        for (std::uint32_t k = 0; k < m.triCount; ++k)
            for (int c = 0; c < 3; ++c)
                if (v.modelTris[m.firstTri + k].v[c] >= m.vertCount) return Fail(error, "model vertex index");
    }
    for (std::uint32_t id : v.modelInstances)
        if (id >= v.instances.size()) return Fail(error, "model instance ref");
    for (std::size_t i = 0; i < v.instances.size(); ++i) {
        const Instance &in = v.instances[i];
        if (in.model >= v.models.size() || in.actor >= v.actors.size() || in.firstTri > nt || in.triCount > nt - in.firstTri ||
            in.triCount != v.models[in.model].triCount)
            return Fail(error, "instance " + std::to_string(i));
    }
    for (const Actor &a : v.actors) {
        if (a.firstInstance > v.instances.size() || a.instanceCount > v.instances.size() - a.firstInstance ||
            (a.trigger != kNoTrigger && a.trigger >= v.triggers.size()))
            return Fail(error, "actor ranges");
    }
    std::vector<std::uint8_t> seen(nt, 0u);
    for (std::size_t t = 0; t < nt; ++t) {
        const TriInfo &x = v.triInfo[t];
        if (x.instance >= v.instances.size() || (v.tris[t].meta >> kMetaActorShift) != v.instances[x.instance].actor ||
            x.modelTri >= v.models[v.instances[x.instance].model].triCount || x.fvOrder >= nt || seen[x.fvOrder] ||
            (x.trigger != kNoTrigger && x.trigger >= v.triggers.size()))
            return Fail(error, "triangle " + std::to_string(t));
        seen[x.fvOrder] = 1u;
    }
    for (const TriggerVolume &tv : v.triggerVolumes)
        if (tv.actor >= v.actors.size() || tv.firstTri > nt || tv.triCount > nt - tv.firstTri) return Fail(error, "trigger volume");
    for (const SpecialSurface &s : v.specials)
        if (s.actor >= v.actors.size()) return Fail(error, "special surface");
    if (v.spawns.empty()) return Fail(error, "no race start spawn");
    if (h.water.enabled && v.waterCells.size() != std::size_t(h.water.dims[0]) * h.water.dims[1] + kWaterCellPadding)
        return Fail(error, "water cells");
    const race::TwRaceMap map = v.RaceMap();
    if (!race::TwRaceMapValid(map)) return Fail(error, "race map limits (TwRaceMapValid)");
    *view = v;
    return true;
}

race::TwRaceMap WorldPackView::RaceMap() const {
    race::TwRaceMap m;
    m.triggers = triggers.ptr;
    m.triggerCount = static_cast<std::uint32_t>(triggers.size());
    m.spawns = spawns.ptr;
    m.spawnCount = static_cast<std::uint32_t>(spawns.size());
    m.requiredCps = header ? header->requiredCps : 0u;
    m.requiredLaps = header ? header->requiredLaps : 1u;
    m.shortcutMode = header && (header->raceFlags & kRaceShortcutMode) != 0u;
    return m;
}

void MakeBenchWorld(const WorldPackView &view, World *out, SourceScene *source) {
    if (source != nullptr) {
        SourceScene src;
        for (const Instance &in : view.instances) {
            const Model &m = view.models[in.model];
            SourceSurface ss{};
            std::memcpy(ss.iso, in.iso, sizeof ss.iso);
            ss.firstVertex = static_cast<std::uint32_t>(src.vertices.size() / 3);
            ss.vertexCount = m.vertCount;
            ss.firstTri = static_cast<std::uint32_t>(src.tris.size());
            ss.triCount = m.triCount;
            ss.actor = in.actor;
            ss.trigger = (in.flags & kInstTrigger) ? 1u : 0u;
            for (std::uint32_t k = 0; k < m.vertCount; ++k) {
                const Vec3f &v = view.modelVerts[m.firstVert + k];
                src.vertices.push_back(v.x);
                src.vertices.push_back(v.y);
                src.vertices.push_back(v.z);
            }
            for (std::uint32_t k = 0; k < m.triCount; ++k) {
                const ModelTri &t = view.modelTris[m.firstTri + k];
                src.tris.push_back(SourceTri{{t.v[0], t.v[1], t.v[2]}, t.meta & kModelTriMaterialMask});
            }
            src.surfaces.push_back(ss);
        }
        *source = std::move(src);
    }
    World w;
    w.tris.assign(view.tris.begin(), view.tris.end());
    w.extra.resize(view.tris.size());
    for (std::size_t i = 0; i < view.tris.size(); ++i) {
        const Tri &t = view.tris[i];
        TriExtra x{};
        const double e1[3] = {double(t.v[1][0]) - t.v[0][0], double(t.v[1][1]) - t.v[0][1], double(t.v[1][2]) - t.v[0][2]};
        const double e2[3] = {double(t.v[2][0]) - t.v[0][0], double(t.v[2][1]) - t.v[0][1], double(t.v[2][2]) - t.v[0][2]};
        const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (l > 1e-12)
            for (int a = 0; a < 3; ++a) x.n[a] = static_cast<float>(n[a] / l);
        x.surface = view.instances[view.triInfo[i].instance].fvSurface;
        w.extra[i] = x;
    }
    const PackHeader &h = *view.header;
    w.actorCount = h.actorCount;
    w.surfaceCount = h.surfaceCount;
    w.triggerTris = h.triggerTris;
    w.skippedSurfaces = h.skippedSurfaces;
    for (int a = 0; a < 3; ++a) {
        w.bounds.lo[a] = h.boundsLo[a];
        w.bounds.hi[a] = h.boundsHi[a];
    }
    *out = std::move(w);
}

}  // namespace tw::world
