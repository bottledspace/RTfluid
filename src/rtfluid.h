#include <optix.h>
#include <vector_functions.h>
#include <vector_types.h>


#define PARTICLE_RADIUS (0.05f)

struct Params {
    unsigned char *image;
    size_t image_width;
    size_t image_height;
    size_t numParticles;
    float3 *pos0;
    float3 *pos1;
    float3 *vel;
    OptixTraversableHandle gas_handle;
};