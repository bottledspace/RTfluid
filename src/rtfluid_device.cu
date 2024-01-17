#include "rtfluid.h"


__constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 d = make_float3(2.0f * (float)idx.x/dim.x - 1.0f,
                                 2.0f * (float)idx.y/dim.y - 1.0f, 1.0f);

    const float3 rayOrigin      = make_float3(0.0f, 0.0f, 0.0f);
    const float3 rayDirection   = normalize(d);
    unsigned hit; // Payload
    optixTrace(
            params.gas_handle,
            rayOrigin,
            rayDirection,
            0.0f,
            1e9f,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            hit);

    unsigned index = 4*(idx.y*params.image_width+idx.x);

    params.image[index+0] = hit*255;
    params.image[index+1] = hit*255;
    params.image[index+2] = hit*255;
    params.image[index+3] = 255;
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0); // no hits
}

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1); // yes a hit
}


extern "C" __global__ void __intersection__sphere()
{
    const unsigned id = optixGetPrimitiveIndex();
    if (id >= params.numParticles)
        return;
    
    const float3 rayOrig = optixGetWorldRayOrigin();
    const float3 rayDir  = optixGetWorldRayDirection();
    const float  rayTmin = optixGetRayTmin();
    const float  rayTmax = optixGetRayTmax();
    const float3 center = params.particles[id].pos;

    const float t = dot(center - rayOrig, rayDir);
    if (t < rayTmin || t > rayTmax)
        return;
    const float3 disp = center - rayDir * t;
    if (dot(disp, disp) <= PARTICLE_RADIUS*PARTICLE_RADIUS)
        optixReportIntersection(t, 0u);
}
