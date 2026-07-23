#ifndef TMNF_GMSURF_COLLISION_H
#define TMNF_GMSURF_COLLISION_H

#include "engine/physics/geometry/gm_surface.h"

int GmCollision_NotHandled(
        const LocatedGmSurf &a,
        const LocatedGmSurf &b,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Sphere_Sphere(
        const LocatedGmSurf &a,
        const LocatedGmSurf &b,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Sphere_Box(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &box,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Sphere_Ellipsoid(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &ellipsoid,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Sphere_Polygon(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &polygon,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Sphere_Mesh(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &mesh,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Sphere_Mesh_OptimizedCpu(
        const LocatedGmSurf &sphere,
        const LocatedGmSurf &mesh,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Ellipsoid_Polygon(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &polygon,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Ellipsoid_Mesh(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &mesh,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Ellipsoid_Mesh_OptimizedCpu(
        const LocatedGmSurf &ellipsoid,
        const LocatedGmSurf &mesh,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Box_Box(
        const LocatedGmSurf &a,
        const LocatedGmSurf &b,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Box_Mesh(
        const LocatedGmSurf &box,
        const LocatedGmSurf &mesh,
        CGmCollisionBuffer &collisionBuffer);
int GmCollision_Mesh_Mesh(
        const LocatedGmSurf &a,
        const LocatedGmSurf &b,
        CGmCollisionBuffer &collisionBuffer);

int GmCoplanarTriTri(
        const GmVec3 &normal,
        const GmVec3 &triA0,
        const GmVec3 &triA1,
        const GmVec3 &triA2,
        const GmVec3 &triB0,
        const GmVec3 &triB1,
        const GmVec3 &triB2);

#endif // TMNF_GMSURF_COLLISION_H
