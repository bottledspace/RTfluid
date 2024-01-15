#include <optix.h>

#define PARTICLE_RADIUS (0.05f)
struct Particle {
    float x,y,z;
};

struct Params {
    unsigned char *image;
    size_t image_width;
    size_t image_height;
    size_t numParticles;
    struct Particle *particles;
    OptixTraversableHandle gas_handle;
};