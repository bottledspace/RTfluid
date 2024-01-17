#include <optix.h>
#include <vector_functions.h>
#include <vector_types.h>
#include "vec_math.h"

#define PARTICLE_RADIUS (0.01f)

#define MODE_UPDATE 1
#define MODE_DRAW   0

struct Particle {
    float3 pos;
    float3 vel;
    int next;
};

struct Params {
    unsigned char *image;
    size_t image_width;
    size_t image_height;
    size_t numParticles;
    struct Particle *particles;
    int *grid;
    OptixTraversableHandle gas_handle;
};