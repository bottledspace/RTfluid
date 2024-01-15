#include "rtfluid.h"
#include "vec_math.h"

__constant__ Params params;

extern "C" __global__ void __raygen__update()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const unsigned id = idx.x + idx.y * dim.x;
    const float speed = length(params.vel[id]);
 
    unsigned p0, p1; // Payload
    optixTrace(
            params.gas_handle,
            params.pos0[id],
            params.vel[id]*(1.0f/speed),
            0.0f,
            speed,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only take the closest hit
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1);
    
    params.pos1[id] = params.pos0[id] + (__uint_as_float(p1)/speed)*params.vel[id];
}

extern "C" __global__ void __raygen__draw()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 d = make_float3(2.0f * (float)idx.x/dim.x - 1.0f,
                                 2.0f * (float)idx.y/dim.y - 1.0f, 1.0f);

    const float3 rayOrigin      = make_float3(0.0f, 0.0f, 0.0f);
    const float3 rayDirection   = normalize(d);
    unsigned p0, p1; // Payload
    optixTrace(
            params.gas_handle,
            rayOrigin,
            rayDirection,
            0.0f,
            1e9f,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1);

    unsigned index = 4*(idx.y*params.image_width+idx.x);

    params.image[index+0] = p0*255;
    params.image[index+1] = p0*255;
    params.image[index+2] = p0*255;
    params.image[index+3] = 255;
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0); // no hits
    optixSetPayload_1(0);
}

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1); // yes a hit
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __intersection__sphere()
{
    int id = optixGetPrimitiveIndex();
    if (id > params.numParticles)
        return;

    const float3 rayOrig = optixGetWorldRayOrigin();
    const float3 rayDir  = optixGetWorldRayDirection();
    const float  rayTmin = optixGetRayTmin();
    const float  rayTmax = optixGetRayTmax();
    const float3 circleCenter = params.pos0[id];

    const float t = dot(circleCenter - rayOrig, rayDir);
    if (t < rayTmin || t > rayTmax)
        return;

    const float3 disp = circleCenter - rayDir * t;
    if (dot(disp, disp) <= PARTICLE_RADIUS * PARTICLE_RADIUS)
        optixReportIntersection(t,0,0u,0u);
}
