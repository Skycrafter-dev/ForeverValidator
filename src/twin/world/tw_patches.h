// Coplanar patches of the world package's models (m2-broadphase.md §6, now over tw_world_pack models).
//
// Per model, adjacent triangles that share an edge with consistent winding, have the same material and the
// same plane (unit normals within angleEps rad, offsets within planeEps m) are merged greedily into convex
// patches. Collinear loop vertices are dropped. A triangle that merges with nothing stays alone (no patch).
// The loop runs counter-clockwise about the normal, as the members' winding, so a loop edge is a member
// edge or a chain of collinear member edges.
//
// Used by the collision stage's "merged" patch semantics (tw_collide.h CollideOptions::patches): a patch is
// one element, its FACE and features come from its loop, and its internal edges make no contacts.
#ifndef TW_PATCHES_H
#define TW_PATCHES_H

#include <cstdint>
#include <vector>

#include "tw_world_pack.h"

namespace tw::patch {

constexpr std::uint32_t kNoPatch = 0xFFFFFFFFu;

struct Patch {
    std::uint32_t firstLoop, loopCount;      // Set::loop vertices (mesh-local xyz), CCW about the normal;
                                             // Set::edge: per loop edge its in-plane outward unit normal and offset
    std::uint32_t firstMember, memberCount;  // Set::members: global ModelTris indices
    std::uint32_t rep;                       // the member with the largest area (global ModelTris index): the plane
    float tau;                               // max |n . v - off| over the members' float vertices (m)
    float alpha;                             // max |n_member - n| over the members (unit normals, double)
};

struct Set {
    std::vector<Patch> patches;
    std::vector<std::uint32_t> patchOf;      // per global ModelTris index: its patch, or kNoPatch
    std::vector<std::uint32_t> members;
    std::vector<float> loop;                 // xyz per loop vertex (exact source floats)
    std::vector<float> edge;                 // per loop edge (v_i -> v_i+1): u (xyz, unit, in-plane outward) and u . v_i
    std::size_t trianglesInPatches = 0;
};

// planeEps (m), angleEps (rad): the coplanarity tolerances (§6: 1e-4 and 1e-4).
void Build(const world::WorldPackView &view, float planeEps, float angleEps, Set *out);

}  // namespace tw::patch

#endif
