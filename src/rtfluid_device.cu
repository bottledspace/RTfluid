#include "rtfluid.h"
#include <optix.h>
#include <optix_device.h>
#include "vec_math.h"

__constant__ Params params;

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd)
{
    unsigned int p0, p1, p2;
    p0 = __float_as_int(prd->x);
    p1 = __float_as_int(prd->y);
    p2 = __float_as_int(prd->z);
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2 );
    prd->x = __int_as_float(p0);
    prd->y = __int_as_float(p1);
    prd->z = __int_as_float(p2);
}


static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_int(p.x));
    optixSetPayload_1(__float_as_int(p.y));
    optixSetPayload_2(__float_as_int(p.z));
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            __int_as_float(optixGetPayload_0()),
            __int_as_float(optixGetPayload_1()),
            __int_as_float(optixGetPayload_2()));
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float3 d = make_float3(2.0f * (float)idx.x/dim.x - 1.0f,
                                 2.0f * (float)idx.y/dim.y - 1.0f, 1.0f);

    const float3 origin      = make_float3(0.0f, 0.0f, 0.0f);
    const float3 direction   = normalize(d);
    float3       payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
    trace(params.gas_handle, origin, direction,
          0.00f, 1e16f, &payload_rgb);

    params.image[4*(idx.y*params.image_width+idx.x)+0] = (uint8_t)(255.0f*payload_rgb.x);
    params.image[4*(idx.y*params.image_width+idx.x)+1] = (uint8_t)(255.0f*payload_rgb.y);
    params.image[4*(idx.y*params.image_width+idx.x)+2] = (uint8_t)(255.0f*payload_rgb.z);
    params.image[4*(idx.y*params.image_width+idx.x)+3] = 255;
}


extern "C" __global__ void __miss__ms()
{
    setPayload(make_float3(0.0f, 1.0f, 0.0f));
}


extern "C" __global__ void __closesthit__ch()
{
    setPayload(make_float3(1.0f, 1.0f, 0.0f));
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
