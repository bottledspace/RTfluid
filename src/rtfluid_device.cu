#include "rtfluid.h"
#include <optix.h>
#include <optix_device.h>
#include "vec_math.h"

__constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 d = make_float3(2.0f * (float)idx.x/dim.x - 1.0f,
                                 2.0f * (float)idx.y/dim.y - 1.0f, 1.0f);

    const float3 rayOrigin      = make_float3(0.0f, 0.0f, 0.0f);
    const float3 rayDirection   = normalize(d);
    unsigned p0, p1, p2; // Payload
    optixTrace(
            params.gas_handle,
            rayOrigin,
            rayDirection,
            1e-3f,
            1e9f,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2);

    unsigned index = 4*(idx.y*params.image_width+idx.x);
    params.image[index+0] = p0;
    params.image[index+1] = p1;
    params.image[index+2] = p2;
    params.image[index+3] = 255;
}


extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0);
    optixSetPayload_1(128);
    optixSetPayload_2(255);
}


extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(255);
    optixSetPayload_1(128);
    optixSetPayload_2(0);
}

extern "C" __global__ void __intersection__sphere()
{
    int index = optixGetPrimitiveIndex();
    if (index > params.numParticles)
        return;

    const float3 rayOrig = optixGetWorldRayOrigin();
    const float3 rayDir  = optixGetWorldRayDirection();
    const float  rayTmin = optixGetRayTmin();
    const float  rayTmax = optixGetRayTmax();
    const float3 circleCenter = make_float3(params.particles[index].x, params.particles[index].y, params.particles[index].z);

    const float t = dot(circleCenter - rayOrig, rayDir);
    const float3 disp = circleCenter - rayDir * t;
    if (dot(disp, disp) <= PARTICLE_RADIUS * PARTICLE_RADIUS)
        optixReportIntersection(t,0,0u,0u);
}
