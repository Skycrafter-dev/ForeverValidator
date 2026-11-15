// Contact atlas: packed column grid (architect C). See atlas_colpack.h.
#include "atlas_colpack.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tw::atlas {

namespace {

template <class T> void Put(std::string &s, T v) { s.append(reinterpret_cast<const char *>(&v), sizeof v); }
template <class T> T Get(const std::uint8_t *&p) { T v; std::memcpy(&v, p, sizeof v); p += sizeof v; return v; }

inline void PutVar(std::string &s, std::uint32_t v) {
    while (v >= 0x80u) { s.push_back(static_cast<char>((v & 0x7Fu) | 0x80u)); v >>= 7; }
    s.push_back(static_cast<char>(v));
}
inline std::uint32_t GetVar(const std::uint8_t *&p) {
    std::uint32_t v = *p & 0x7Fu;
    if (!(*p++ & 0x80u)) return v;
    int sh = 7;
    for (;;) {
        const std::uint8_t b = *p++;
        v |= std::uint32_t(b & 0x7Fu) << sh;
        if (!(b & 0x80u)) return v;
        sh += 7;
    }
}

constexpr std::uint8_t kFullBox[4] = {0, 255, 0, 255};

}  // namespace

void ColumnPack::Build(const RawColumns &raw) {
    x0_ = raw.x0; z0_ = raw.z0; nx_ = raw.nx; nz_ = raw.nz;
    sx_ = (nx_ + kSuper - 1) / kSuper;
    sz_ = (nz_ + kSuper - 1) / kSuper;
    supers_.assign(std::size_t(sx_) * sz_, Super{});
    plateIds_.clear();
    bytes_.clear();
    plateFirstMember_.resize(raw.plateCount);
    plateWideMembers_.resize(raw.plateCount);
    for (std::size_t p = 0; p < raw.plateCount; ++p) {
        plateFirstMember_[p] = raw.plates[p].firstMember;
        plateWideMembers_[p] = raw.plates[p].memberCount > 65536 ? 1 : 0;
    }
    const std::uint32_t slotsPerSuper = kSuper * kSuper * 2;
    std::vector<std::string> pays(slotsPerSuper);
    for (std::uint32_t bz = 0; bz < sz_; ++bz)
        for (std::uint32_t bx = 0; bx < sx_; ++bx) {
            Super &sb = supers_[std::size_t(bz) * sx_ + bx];
            auto slotOf = [&](std::uint32_t lz, std::uint32_t lx, int cls) -> std::int64_t {
                const std::uint32_t z = bz * kSuper + lz, x = bx * kSuper + lx;
                if (z >= nz_ || x >= nx_) return -1;
                return std::int64_t((std::size_t(z) * nx_ + x) * 2 + std::size_t(cls));
            };
            std::unordered_map<std::uint32_t, std::uint32_t> local;
            std::vector<std::uint32_t> ids;
            bool any = false;
            for (std::uint32_t lz = 0; lz < kSuper; ++lz)
                for (std::uint32_t lx = 0; lx < kSuper; ++lx)
                    for (int cls = 0; cls < 2; ++cls) {
                        const std::int64_t s = slotOf(lz, lx, cls);
                        if (s < 0) continue;
                        for (std::uint32_t i = raw.first[s]; i < raw.first[s + 1]; ++i) {
                            if (!any) sb.yBase = raw.entries[i].y0;
                            any = true;
                            if (local.emplace(raw.entries[i].plate, std::uint32_t(ids.size())).second) ids.push_back(raw.entries[i].plate);
                        }
                    }
            if (!any) continue;   // all slots empty (plateCount 0: no payload)
            if (ids.size() > 65535) throw std::runtime_error("colpack: more than 65535 plates in a superblock");
            sb.plates = static_cast<std::uint32_t>(plateIds_.size());
            sb.plateCount = static_cast<std::uint16_t>(ids.size());
            plateIds_.insert(plateIds_.end(), ids.begin(), ids.end());
            sb.wideIdx = ids.size() > 256 ? 1 : 0;
            for (std::uint32_t lz = 0; lz < kSuper; ++lz)
                for (std::uint32_t lx = 0; lx < kSuper; ++lx)
                    for (int cls = 0; cls < 2; ++cls) {
                        std::string &pay = pays[(lz * kSuper + lx) * 2 + std::uint32_t(cls)];
                        pay.clear();
                        const std::int64_t s = slotOf(lz, lx, cls);
                        if (s < 0) continue;
                        const std::uint32_t f = raw.first[s], l = raw.first[s + 1];
                        if (f == l) continue;
                        // SoA, fixed width: n, flags, base, y0 offsets, y1 - y0, plate indices, box mask + boxes
                        const std::uint32_t n = l - f;
                        const std::int32_t base = raw.entries[f].y0;
                        bool wideY = false, anyBox = false;
                        for (std::uint32_t i = f; i < l; ++i) {
                            const ColumnEntry &e = raw.entries[i];
                            wideY = wideY || std::int64_t(e.y0) - base > 65535 || std::int64_t(e.y1) - e.y0 > 65535 || std::int64_t(e.y1max) - base > 65535;
                            anyBox = anyBox || std::memcmp(e.box, kFullBox, 4) != 0;
                        }
                        if (n < 255) Put<std::uint8_t>(pay, static_cast<std::uint8_t>(n)); else { Put<std::uint8_t>(pay, 255); Put<std::uint32_t>(pay, n); }
                        Put<std::uint8_t>(pay, static_cast<std::uint8_t>((wideY ? 1 : 0) | (anyBox ? 2 : 0)));
                        Put<std::int32_t>(pay, base);
                        for (std::uint32_t i = f; i < l; ++i) {
                            if (wideY) Put<std::uint32_t>(pay, std::uint32_t(raw.entries[i].y0 - base));
                            else Put<std::uint16_t>(pay, static_cast<std::uint16_t>(raw.entries[i].y0 - base));
                        }
                        for (std::uint32_t i = f; i < l; ++i) {   // E's prefix max of y1: the free test's search key
                            if (wideY) Put<std::uint32_t>(pay, std::uint32_t(raw.entries[i].y1max - base));
                            else Put<std::uint16_t>(pay, static_cast<std::uint16_t>(raw.entries[i].y1max - base));
                        }
                        for (std::uint32_t i = f; i < l; ++i) {
                            if (wideY) Put<std::uint32_t>(pay, std::uint32_t(raw.entries[i].y1 - raw.entries[i].y0));
                            else Put<std::uint16_t>(pay, static_cast<std::uint16_t>(raw.entries[i].y1 - raw.entries[i].y0));
                        }
                        for (std::uint32_t i = f; i < l; ++i) {
                            const std::uint32_t li = local[raw.entries[i].plate];
                            if (sb.wideIdx) Put<std::uint16_t>(pay, static_cast<std::uint16_t>(li)); else Put<std::uint8_t>(pay, static_cast<std::uint8_t>(li));
                        }
                        if (anyBox) {
                            std::string mask((n + 7) / 8, '\0'), boxes;
                            for (std::uint32_t i = f; i < l; ++i)
                                if (std::memcmp(raw.entries[i].box, kFullBox, 4) != 0) {
                                    mask[(i - f) / 8] = static_cast<char>(mask[(i - f) / 8] | (1 << ((i - f) % 8)));
                                    boxes.append(reinterpret_cast<const char *>(raw.entries[i].box), 4);
                                }
                            pay += mask;
                            pay += boxes;
                        }
                        for (std::uint32_t i = f; i < l; ++i) {
                            const ColumnEntry &e = raw.entries[i];
                            const std::uint32_t m0 = raw.entMemFirst[i], m1 = raw.entMemFirst[i + 1];
                            PutVar(pay, m1 - m0);
                            if (m1 == m0) { if (raw.entOverlap[i]) throw std::runtime_error("colpack: overlap flag without members"); continue; }
                            Put<std::uint8_t>(pay, raw.entOverlap[i]);
                            const bool wide = e.plate == kLoose || plateWideMembers_[e.plate];
                            const std::uint32_t base = e.plate == kLoose ? 0 : plateFirstMember_[e.plate];
                            for (std::uint32_t m = m0; m < m1; ++m) {
                                if (wide) Put<std::uint32_t>(pay, raw.entMembers[m]);
                                else Put<std::uint16_t>(pay, static_cast<std::uint16_t>(raw.entMembers[m] - base));
                            }
                        }
                    }
            // offsets, then the deduplicated payloads
            std::unordered_map<std::string, std::uint32_t> seen;
            std::vector<std::uint32_t> off(slotsPerSuper, 0xFFFFFFFFu);
            std::string data;
            for (std::uint32_t q = 0; q < slotsPerSuper; ++q) {
                if (pays[q].empty()) continue;
                auto it = seen.find(pays[q]);
                if (it != seen.end()) { off[q] = it->second; continue; }
                off[q] = static_cast<std::uint32_t>(data.size());
                seen.emplace(pays[q], off[q]);
                data += pays[q];
            }
            sb.wideOff = data.size() >= 0xFFFF ? 1 : 0;
            std::string blob;
            for (std::uint32_t q = 0; q < slotsPerSuper; ++q) {
                if (sb.wideOff) Put<std::uint32_t>(blob, off[q]);
                else Put<std::uint16_t>(blob, off[q] == 0xFFFFFFFFu ? std::uint16_t(0xFFFF) : static_cast<std::uint16_t>(off[q]));
            }
            blob += data;
            if (bytes_.size() + blob.size() > 0xFFFFFFFFull) throw std::runtime_error("colpack: arena over 4 GB");
            sb.data = static_cast<std::uint32_t>(bytes_.size());
            bytes_.insert(bytes_.end(), blob.begin(), blob.end());
        }
    bytes_.shrink_to_fit();
}

std::size_t ColumnPack::Bytes() const {
    return bytes_.size() + supers_.size() * sizeof(Super) + plateIds_.size() * 4 + plateFirstMember_.size() * 5;
}

const std::uint8_t *ColumnPack::Payload(std::uint32_t handle, const Super **sbOut) const {
    const Super &sb = supers_[handle >> 9];
    *sbOut = &sb;
    if (sb.plateCount == 0) return nullptr;
    const std::uint32_t q = handle & 511u;
    const std::uint8_t *base = bytes_.data() + sb.data;
    constexpr std::uint32_t slots = kSuper * kSuper * 2;
    if (sb.wideOff) {
        std::uint32_t off;
        std::memcpy(&off, base + 4 * q, 4);
        return off == 0xFFFFFFFFu ? nullptr : base + 4 * slots + off;
    }
    std::uint16_t o16;
    std::memcpy(&o16, base + 2 * q, 2);
    return o16 == 0xFFFF ? nullptr : base + 2 * slots + o16;
}

void ColumnPack::Decode(std::uint32_t handle, SlotView *out, int what) const {
    out->n = 0;
    out->unionCount = 0;
    out->entries.clear();
    out->unionYs.clear();
    out->members.clear();
    out->overlap.clear();
    out->memFirst.assign(1, 0);
    const Super *sb;
    const std::uint8_t *p = Payload(handle, &sb);
    if (!p) return;
    std::uint32_t n = *p++;
    if (n == 255) n = Get<std::uint32_t>(p);
    out->n = n;
    out->entries.resize(n);
    const std::uint8_t flags = *p++;
    const std::int32_t base = Get<std::int32_t>(p);
    const bool wideY = flags & 1u, anyBox = flags & 2u;
    const std::uint8_t *py0 = p, *pdy = p + 2 * (wideY ? 4 : 2) * n, *pidx = pdy + (wideY ? 4 : 2) * n;
    const std::uint8_t *pmask = pidx + (sb->wideIdx ? 2 : 1) * n;
    const std::uint8_t *pbox = pmask + (anyBox ? (n + 7) / 8 : 0);
    std::int32_t m = INT32_MIN;
    for (std::uint32_t i = 0; i < n; ++i) {
        ColumnEntry &e = out->entries[i];
        std::uint32_t o, d, li;
        if (wideY) { std::memcpy(&o, py0 + 4 * i, 4); std::memcpy(&d, pdy + 4 * i, 4); }
        else { std::uint16_t a16, b16; std::memcpy(&a16, py0 + 2 * i, 2); std::memcpy(&b16, pdy + 2 * i, 2); o = a16; d = b16; }
        if (sb->wideIdx) { std::uint16_t l16; std::memcpy(&l16, pidx + 2 * i, 2); li = l16; } else li = pidx[i];
        e.plate = plateIds_[sb->plates + li];
        e.y0 = std::int32_t(std::uint32_t(base) + o);
        e.y1 = std::int32_t(std::uint32_t(e.y0) + d);
        m = std::max(m, e.y1);   // E: m = std::max(m, y1), the prefix max (equal to the stored one)
        e.y1max = m;
        if (anyBox && (pmask[i / 8] >> (i % 8) & 1u)) { std::memcpy(e.box, pbox, 4); pbox += 4; }
        else std::memcpy(e.box, kFullBox, 4);
    }
    p = pbox;
    if (what & 2) {   // E's union merge over the sorted entries
        std::int32_t u0 = 0, u1 = 0;
        bool open = false;
        for (std::uint32_t q = 0; q < n; ++q) {
            const ColumnEntry &e = out->entries[q];
            if (!open) { u0 = e.y0; u1 = e.y1; open = true; continue; }
            if (e.y0 <= u1) u1 = std::max(u1, e.y1);
            else { out->unionYs.push_back(u0); out->unionYs.push_back(u1); u0 = e.y0; u1 = e.y1; }
        }
        if (open) { out->unionYs.push_back(u0); out->unionYs.push_back(u1); }
        out->unionCount = static_cast<std::uint32_t>(out->unionYs.size() / 2);
    }
    if (what & 4) {
        out->memFirst.assign(n + 1, 0);
        out->overlap.assign(n, 0);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t c = GetVar(p);
            out->memFirst[i] = static_cast<std::uint32_t>(out->members.size());
            if (c == 0) continue;
            out->overlap[i] = *p++;
            const ColumnEntry &e = out->entries[i];
            const bool wide = e.plate == kLoose || plateWideMembers_[e.plate];
            const std::uint32_t base = e.plate == kLoose ? 0 : plateFirstMember_[e.plate];
            for (std::uint32_t k = 0; k < c; ++k) out->members.push_back(wide ? Get<std::uint32_t>(p) : base + Get<std::uint16_t>(p));
        }
        out->memFirst[n] = static_cast<std::uint32_t>(out->members.size());
    }
}

// Entries are sorted by y0 with the prefix max y1max: the first entry whose y1max reaches lo is the first
// with y1 >= lo, and some entry meets [lo, hi] exactly when that entry's y0 <= hi (later ones start higher,
// earlier ones end below lo).
bool ColumnPack::AnyOverlap(std::uint32_t handle, std::int32_t lo, std::int32_t hi) const {
    const Super *sb;
    const std::uint8_t *p = Payload(handle, &sb);
    if (!p) return false;
    std::uint32_t n = *p++;
    if (n == 255) n = Get<std::uint32_t>(p);
    const std::uint8_t flags = *p++;
    const std::int32_t base = Get<std::int32_t>(p);
    const std::int64_t rlo = std::int64_t(lo) - base, rhi = std::int64_t(hi) - base;
    if (rhi < 0) return false;   // every entry starts at or above base
    if (flags & 1u) {
        const std::uint8_t *pm = p + 4 * n;
        std::uint32_t a = 0, b = n;   // first i with y1max >= rlo
        while (a < b) {
            const std::uint32_t c = (a + b) / 2;
            std::uint32_t v;
            std::memcpy(&v, pm + 4 * c, 4);
            if (std::int64_t(v) >= rlo) b = c; else a = c + 1;
        }
        if (a == n) return false;
        std::uint32_t y0;
        std::memcpy(&y0, p + 4 * a, 4);
        return std::int64_t(y0) <= rhi;
    }
    const std::uint16_t *y0 = reinterpret_cast<const std::uint16_t *>(p);
    const std::uint16_t *ym = reinterpret_cast<const std::uint16_t *>(p + 2 * n);
    if (rlo <= 0) return true;   // the first entry starts at base <= hi and ends at or above base >= lo... (y1 >= y0 = base >= lo)
    if (rlo > 65535) return false;   // y1max <= 65535 + base < lo
    const std::uint16_t key = static_cast<std::uint16_t>(rlo);
    std::uint32_t a = 0;
    for (; a < n; ++a) {   // y1max is non-decreasing and short: linear
        std::uint16_t v;
        std::memcpy(&v, ym + a, 2);
        if (v >= key) break;
    }
    if (a == n) return false;
    std::uint16_t v0;
    std::memcpy(&v0, y0 + a, 2);
    return v0 <= rhi;
}

}  // namespace tw::atlas
