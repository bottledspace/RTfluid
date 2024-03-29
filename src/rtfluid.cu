#include "rtfluid.h"
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>
#include <windows.h>

//extern unsigned char rtfluid_device[];
//extern uint32_t rtfluid_deviceLength;
// TODO: Fix linker errors so we don't need to do this
#include "../build/rtfluid_device_ptx.cu"

#define ALIGN_SIZE(x) (((size_t)(x) + 7) & ~7u)

#define LOG_FUNCTION() //printf("[%s:%d]: Enter\n", __FILE__, __LINE__)

#define CUDA_CHECK(func) cudaCheckImpl((func), __FILE__, __LINE__)
static void cudaCheckImpl(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
    //fprintf(stderr, "[%s:%d]: OK\n", file, line);
}

#define OPTIX_CHECK(func) optixCheckImpl((func), __FILE__, __LINE__)
static void optixCheckImpl(OptixResult code, const char *file, int line)
{
    if (code != OPTIX_SUCCESS) {
        fprintf(stderr, "[%s:%d]: %s\n", file, line, optixGetErrorString(code));
        exit(code);
    }
    //fprintf(stderr, "[%s:%d]: OK\n", file, line);
}

struct RayGenSbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct MissSbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct HitGroupSbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

char logBuffer[2048];
size_t logSize = 2048;

const int width  = 800;
const int height = 600;
const int comp   = 4;

#define GRID_SIZE 100

struct Params               params;
OptixDeviceContext          context;
OptixTraversableHandle      gas_handle;
CUdeviceptr                 d_gas_output_buffer;
OptixModule                 module;
OptixPipelineCompileOptions pipeline_compile_options;
OptixProgramGroup           raygen_prog_group;
OptixProgramGroup           miss_prog_group;
OptixProgramGroup           hitgroup_prog_group;
OptixPipeline               pipeline;
OptixShaderBindingTable     sbt;
GLFWwindow                 *window;
GLuint                      fbID;
uint8_t                    *fb;
cudaGraphicsResource_t      fbRes;
cudaArray_t                 array;
OptixAccelBufferSizes       gas_buffer_sizes;
CUdeviceptr                 d_temp_buffer_gas;
CUdeviceptr                 d_buffer_temp_output_gas_and_compacted_size;
OptixAabb                  *aabbBuffer;


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    fprintf(stderr, "[%d][%s]: %s\n", level, tag, message);
}

static void setupOptix(void)
{
    LOG_FUNCTION();

    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {0};
    memset(&options, 0, sizeof(options));
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK(optixDeviceContextCreate(0 /*=> take current context */, &options, &context));
}

__global__ void adjustAABBs(int numParticles, struct Particle *particles, OptixAabb* aabb)
{
    int particleIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (particleIdx >= numParticles)
        return;

    const struct Particle *particle = &particles[particleIdx];
    aabb[particleIdx].minX = particle->pos.x - PARTICLE_RADIUS;
    aabb[particleIdx].minY = particle->pos.y - PARTICLE_RADIUS;
    aabb[particleIdx].minZ = particle->pos.z - PARTICLE_RADIUS;
    aabb[particleIdx].maxX = particle->pos.x + PARTICLE_RADIUS;
    aabb[particleIdx].maxY = particle->pos.y + PARTICLE_RADIUS;
    aabb[particleIdx].maxZ = particle->pos.z + PARTICLE_RADIUS;

    // Now is a good time to reset this
    particles[particleIdx].next = -1;
}

__host__ __device__ __forceinline__ static unsigned hashPosition(float3 pos)
{
    int xi = GRID_SIZE*(pos.x+1.0f)/2.0f;
    int yi = GRID_SIZE*(pos.y+1.0f)/2.0f;
    int zi = GRID_SIZE*((pos.z-2.0f)+1.0f)/2.0f;
    return clamp(zi,1,GRID_SIZE-1)*GRID_SIZE*GRID_SIZE + clamp(yi,1,GRID_SIZE-1)*GRID_SIZE + clamp(xi,1,GRID_SIZE-1);
}

__global__ void integrateParticle(int numParticles, struct Particle *particles, int *grid)
{
    int particleIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (particleIdx >= numParticles)
        return;

    struct Particle *particle = &particles[particleIdx];
    if (particle->pos.x > 1.0f) {
        particle->pos.x = 1.0f;
        //particle->vel.x *= -0.01f;
    }
    if (particle->pos.x < -1.0f) {
        particle->pos.x = -1.0f;
        //particle->vel.x *= -0.01f;
    }
    if (particle->pos.y > 1.0f) {
        particle->pos.y = 1.0f;
        //particle->vel.y *= -0.01f;
    }
    if (particle->pos.y < -1.0f) {
        particle->pos.y = -1.0f;
        //particle->vel.y *= -0.01f;
    }
    if (particle->pos.z > 2.0f+1.0f) {
        particle->pos.z = 2.0f+1.0f;
        //particle->vel.z *= -0.01f;
    }
    if (particle->pos.z < 2.0f-1.0f) {
        particle->pos.z = 2.0f-1.0f;
        //particle->vel.z *= -0.01f;
    }
    particle->pos += particle->vel;
    particle->vel *= 0.9f;
    particle->vel.y -= 0.01f;
    particle->next = atomicExch(&grid[hashPosition(particle->pos)], particleIdx);
}



__global__ void fixCollisions(struct Particle *particles, int *grid, int step, dim3 offset)
{
    
    const int3 neighborhoodOffsets[27] = {
        make_int3((-1), (-1), (-1)),
        make_int3((-1), (-1), ( 0)),
        make_int3((-1), (-1), (+1)),
        make_int3((-1), ( 0), (-1)),
        make_int3((-1), ( 0), ( 0)),
        make_int3((-1), ( 0), (+1)),
        make_int3((-1), (+1), (-1)),
        make_int3((-1), (+1), ( 0)),
        make_int3((-1), (+1), (+1)),
        make_int3(( 0), (-1), (-1)),
        make_int3(( 0), (-1), ( 0)),
        make_int3(( 0), (-1), (+1)),
        make_int3(( 0), ( 0), (-1)),
        make_int3(( 0), ( 0), ( 0)),
        make_int3(( 0), ( 0), (+1)),
        make_int3(( 0), (+1), (-1)),
        make_int3(( 0), (+1), ( 0)),
        make_int3(( 0), (+1), (+1)),
        make_int3((+1), (-1), (-1)),
        make_int3((+1), (-1), ( 0)),
        make_int3((+1), (-1), (+1)),
        make_int3((+1), ( 0), (-1)),
        make_int3((+1), ( 0), ( 0)),
        make_int3((+1), ( 0), (+1)),
        make_int3((+1), (+1), (-1)),
        make_int3((+1), (+1), ( 0)),
        make_int3((+1), (+1), (+1)),
    };
    const int zi = (threadIdx.z+blockIdx.z*blockDim.z)*step+offset.z;
    const int yi = (threadIdx.y+blockIdx.y*blockDim.y)*step+offset.y;
    const int xi = (threadIdx.x+blockIdx.x*blockDim.x)*step+offset.x;
    if (xi >= GRID_SIZE || yi >= GRID_SIZE || zi >= GRID_SIZE)
        return;

    const int gridIdx1 = GRID_SIZE*GRID_SIZE*zi + GRID_SIZE*yi + xi;
    for (int offsetIdx2 = 0; offsetIdx2 < 27; offsetIdx2++) {
        int3 neighborhoodOffset = neighborhoodOffsets[offsetIdx2];
        if (neighborhoodOffset.x+xi < 0 || neighborhoodOffset.x+xi >= GRID_SIZE
         || neighborhoodOffset.y+yi < 0 || neighborhoodOffset.y+yi >= GRID_SIZE
         || neighborhoodOffset.z+zi < 0 || neighborhoodOffset.z+zi >= GRID_SIZE)
            continue;
        int gridIdx2 = gridIdx1 + neighborhoodOffset.x + neighborhoodOffset.y*GRID_SIZE + neighborhoodOffset.z*GRID_SIZE*GRID_SIZE;
        for (int particleIdx2 = grid[gridIdx2]; particleIdx2 != -1; particleIdx2 = particles[particleIdx2].next)
        for (int particleIdx1 = grid[gridIdx1]; particleIdx1 != -1; particleIdx1 = particles[particleIdx1].next) {
            if (particleIdx1 == particleIdx2)
                continue;
            
            float3 disp = particles[particleIdx1].pos - particles[particleIdx2].pos;
            float dist2 = dot(disp, disp);
            if (dist2 < 4.0f*(PARTICLE_RADIUS)*(PARTICLE_RADIUS)) {
                float dist = sqrtf(dist2);
                particles[particleIdx1].vel += 0.1f*(PARTICLE_RADIUS)*disp/dist;
                //particles[particleIdx2].vel -= 0.1f*(PARTICLE_RADIUS)*disp/dist;    
            }
        }
    }
}

static void generateParticles(void)
{
    params.numParticles = 102400;

    CUDA_CHECK(cudaMallocManaged((void**)&params.particles, ALIGN_SIZE(sizeof(struct Particle)*params.numParticles)));    
    
    //struct Particle *particles = (struct Particle*)malloc(sizeof(struct Particle)*params.numParticles);
    for (int i = 0; i < params.numParticles; i++) {
        params.particles[i].pos.x = 0.5*((2.0*(float)rand()/(float)RAND_MAX)-1.0f);
        params.particles[i].pos.y = 0.5*((2.0*(float)rand()/(float)RAND_MAX)-1.0f);
        params.particles[i].pos.z = 0.5*((2.0*(float)rand()/(float)RAND_MAX)-1.0f)+2.0f;

        params.particles[i].vel.x =  0.00; //1f*((2.0*(float)rand()/(float)RAND_MAX)-1.0f);
        params.particles[i].vel.y = -0.01; //1f*((2.0*(float)rand()/(float)RAND_MAX)-1.0f);
        params.particles[i].vel.z =  0.00; //1f*((2.0*(float)rand()/(float)RAND_MAX)-1.0f);
    }
    //CUDA_CHECK(cudaMemcpy(params.particles, particles, sizeof(struct Particle)*params.numParticles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocManaged((void**)&params.grid, GRID_SIZE*GRID_SIZE*GRID_SIZE*sizeof(int)));
    CUDA_CHECK(cudaMemset(params.grid, -1, GRID_SIZE*GRID_SIZE*GRID_SIZE*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&aabbBuffer, ALIGN_SIZE(params.numParticles*sizeof(OptixAabb))));
    adjustAABBs<<<params.numParticles/128, 128>>>(params.numParticles, params.particles, aabbBuffer);
    
    //free(particles);
}

static void recreateGAS(OptixBuildOperation operation)
{
    LOG_FUNCTION();

    OptixAccelBuildOptions accel_options;
    memset(&accel_options, 0, sizeof(accel_options));
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    accel_options.operation  = operation;
    
    CUdeviceptr buffers[] = { (CUdeviceptr)aabbBuffer };
    OptixBuildInput aabb_input;
    memset(&aabb_input, 0, sizeof(aabb_input));
    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = buffers;
    aabb_input.customPrimitiveArray.numPrimitives = params.numParticles;

    uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    // Allocate buffers if we haven't yet
    if (operation == OPTIX_BUILD_OPERATION_BUILD) {
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &aabb_input, 1, &gas_buffer_sizes));
        CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer_gas, ALIGN_SIZE(gas_buffer_sizes.tempSizeInBytes)));
        CUDA_CHECK(cudaMalloc((void**)&d_buffer_temp_output_gas_and_compacted_size, ALIGN_SIZE(gas_buffer_sizes.outputSizeInBytes)));
    }
    OPTIX_CHECK(optixAccelBuild(context,
                    0,
                    &accel_options,
                    &aabb_input,
                    1, d_temp_buffer_gas,
                    gas_buffer_sizes.tempSizeInBytes,
                    d_buffer_temp_output_gas_and_compacted_size,
                    gas_buffer_sizes.outputSizeInBytes,
                    &gas_handle,
                    NULL, 0));


}

static void createModule(void)
{
    LOG_FUNCTION();

    OptixModuleCompileOptions module_compile_options;
    memset(&module_compile_options, 0, sizeof(module_compile_options));
/*#if 1
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif*/
    
    memset(&pipeline_compile_options, 0, sizeof(OptixPipelineCompileOptions));
    pipeline_compile_options.usesMotionBlur        = 0;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues      = 1;
    pipeline_compile_options.numAttributeValues    = 1;
    pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK(optixModuleCreate(context, &module_compile_options, &pipeline_compile_options,
                      (const char *)rtfluid_device, rtfluid_deviceLength, logBuffer, &logSize, &module));
}

static void createProgramGroups(void)
{
    LOG_FUNCTION();

    OptixProgramGroupOptions program_group_options;
    memset(&program_group_options, 0, sizeof(OptixProgramGroupOptions));

    LOG_FUNCTION();

    OptixProgramGroupDesc raygen_prog_group_desc;
    memset(&raygen_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygen_prog_group_desc, 1, &program_group_options,
                            logBuffer, &logSize, &raygen_prog_group));
    
    LOG_FUNCTION();

    OptixProgramGroupDesc miss_prog_group_desc;
    memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(context, &miss_prog_group_desc, 1, &program_group_options,
                            logBuffer, &logSize, &miss_prog_group));

    LOG_FUNCTION();

    OptixProgramGroupDesc hitgroup_prog_group_desc;
    memset(&hitgroup_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH            = NULL;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = NULL;
    hitgroup_prog_group_desc.hitgroup.moduleIS            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    
    LOG_FUNCTION();

    OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroup_prog_group_desc, 1, &program_group_options,
                            logBuffer, &logSize, &hitgroup_prog_group));
}

static void createPipeline(void)
{
    LOG_FUNCTION();

    const uint32_t max_trace_depth  = 16;
    OptixProgramGroup program_groups[] = {
        raygen_prog_group,
        miss_prog_group,
        hitgroup_prog_group
    };
    const size_t numProgramGroups = 3;

    OptixPipelineLinkOptions pipeline_link_options;
    memset(&pipeline_link_options, 0, sizeof(OptixPipelineLinkOptions));
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    OPTIX_CHECK(optixPipelineCreate(context, &pipeline_compile_options, &pipeline_link_options,
                        program_groups, numProgramGroups, logBuffer, &logSize, &pipeline));

    OptixStackSizes stack_sizes;
    memset(&stack_sizes, 0, sizeof(OptixStackSizes));
    for (size_t i = 0; i < numProgramGroups; i++)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(program_groups[i], &stack_sizes, pipeline));

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0, &direct_callable_stack_size_from_traversal,
                               &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                              direct_callable_stack_size_from_state, continuation_stack_size, 1));
}

static void createSBT(void)
{
    LOG_FUNCTION();

    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(struct RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc((void**)&raygen_record, raygen_record_size));

    struct RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy((void*)raygen_record, &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));


    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof(struct MissSbtRecord);
    CUDA_CHECK(cudaMalloc((void**)&miss_record, miss_record_size));
    
    struct MissSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy((void*)miss_record, &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));


    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof(struct HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc((void**)&hitgroup_record, hitgroup_record_size));
    
    struct HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy((void*)hitgroup_record, &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));


    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof(struct MissSbtRecord);
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(struct HitGroupSbtRecord);
    sbt.hitgroupRecordCount         = 1;
}

static void setupGUI(void)
{
    assert(glfwInit());
    
    window = glfwCreateWindow(width, height, "RTfluid", NULL, NULL);
    glfwMakeContextCurrent(window);

    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &fbID);
    glBindTexture(GL_TEXTURE_2D, fbID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    CUDA_CHECK(cudaGraphicsGLRegisterImage(&fbRes, fbID, GL_TEXTURE_2D, 0));
    CUDA_CHECK(cudaMallocManaged((void**)&fb, comp*width*height, cudaMemAttachGlobal));
}

CUdeviceptr d_param;

static void launch(void)
{
    LOG_FUNCTION();

    params.image        = fb;
    params.image_width  = width;
    params.image_height = height;
    params.gas_handle   = gas_handle;

    LARGE_INTEGER frequency;        // ticks per second
    LARGE_INTEGER t1, t2;           // ticks

    CUDA_CHECK(cudaDeviceSynchronize());
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    CUDA_CHECK(cudaMemcpy((void*)d_param, &params, sizeof(params), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(pipeline, 0, d_param, sizeof(struct Params), &sbt,
                params.image_width, params.image_height, 1));

    // Transfer from CUDA to OpenGL

    cudaMemcpy2DToArray(array, 0, 0, fb, width * sizeof(uint32_t),
        width * sizeof(uint32_t), height, cudaMemcpyDeviceToDevice);

    CUDA_CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&t2);
    fprintf(stderr, "draw: %lf\n", (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart);

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    integrateParticle<<<params.numParticles/128, 128>>>(params.numParticles, params.particles, params.grid);

#if 1
// Parallel spatial hashing
    fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 1, dim3(0,0,0));

    /*fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 2, dim3(0,0,1));
    fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 2, dim3(1,0,0));

    fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 2, dim3(0,0,0));
    fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 2, dim3(1,1,1));


    fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 2, dim3(0,1,1));
    fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 2, dim3(1,1,0));

    fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 2, dim3(0,1,0));
    fixCollisions<<<dim3(GRID_SIZE/8,GRID_SIZE/8,GRID_SIZE/8), dim3(8,8,8)>>>(params.particles, params.grid, 2, dim3(1,0,1));*/
#endif

#if 0
// Serial spatial hashing (slow) for comparison
    {
        CUDA_CHECK(cudaDeviceSynchronize());

        const int neighborhoodOffsets[27] = {
            (-1)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + (-1),
            (-1)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + ( 0),
            (-1)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + (+1),
            (-1)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + (-1),
            (-1)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + ( 0),
            (-1)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + (+1),
            (-1)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + (-1),
            (-1)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + ( 0),
            (-1)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + (+1),
            ( 0)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + (-1),
            ( 0)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + ( 0),
            ( 0)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + (+1),
            ( 0)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + (-1),
            ( 0)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + ( 0),
            ( 0)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + (+1),
            ( 0)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + (-1),
            ( 0)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + ( 0),
            ( 0)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + (+1),
            (+1)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + (-1),
            (+1)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + ( 0),
            (+1)*GRID_SIZE*GRID_SIZE + (-1)*GRID_SIZE + (+1),
            (+1)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + (-1),
            (+1)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + ( 0),
            (+1)*GRID_SIZE*GRID_SIZE + ( 0)*GRID_SIZE + (+1),
            (+1)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + (-1),
            (+1)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + ( 0),
            (+1)*GRID_SIZE*GRID_SIZE + (+1)*GRID_SIZE + (+1),
        };

        for (int zi = 1; zi < GRID_SIZE-1; zi++)
        for (int yi = 1; yi < GRID_SIZE-1; yi++)
        for (int xi = 1; xi < GRID_SIZE-1; xi++) {
            const int gridIdx = xi+yi*GRID_SIZE+zi*GRID_SIZE*GRID_SIZE;

            for (int offsetIdx2 = 0; offsetIdx2 < 27; offsetIdx2++)
            for (int particleIdx2 = params.grid[neighborhoodOffsets[offsetIdx2]+gridIdx]; particleIdx2 != -1; particleIdx2 = params.particles[particleIdx2].next)
            for (int particleIdx1 = params.grid[                                gridIdx]; particleIdx1 != -1; particleIdx1 = params.particles[particleIdx1].next) {
                if (particleIdx1 == particleIdx2)
                    continue;
                
                float3 disp = params.particles[particleIdx1].pos - params.particles[particleIdx2].pos;
                float dist2 = dot(disp, disp);
                if (dist2 < 4.0f*(PARTICLE_RADIUS)*(PARTICLE_RADIUS)) {
                    float dist = sqrtf(dist2);
                    params.particles[particleIdx1].pos += 0.1f*(PARTICLE_RADIUS)*disp/dist;
                    params.particles[particleIdx2].pos -= 0.1f*(PARTICLE_RADIUS)*disp/dist;
                }
            }
        }
    }
#endif

#if 0
// Naiive (very slow) collision response for comparison
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int particleIdx2 = 0; particleIdx2 < params.numParticles; particleIdx2++)
    for (int particleIdx1 = particleIdx2+1; particleIdx1 < params.numParticles; particleIdx1++) {
        
        float3 disp = params.particles[particleIdx1].pos - params.particles[particleIdx2].pos;
        float dist2 = dot(disp, disp);
        if (dist2 < 4.0f*(PARTICLE_RADIUS)*(PARTICLE_RADIUS)) {
            float dist = sqrtf(dist2);
            params.particles[particleIdx1].pos += 0.01f*(PARTICLE_RADIUS)*disp/dist;
            params.particles[particleIdx2].pos -= 0.01f*(PARTICLE_RADIUS)*disp/dist;
        }
    }
#endif 

    CUDA_CHECK(cudaMemset(params.grid, -1, GRID_SIZE*GRID_SIZE*GRID_SIZE*sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&t2);
    fprintf(stderr, "update: %lf\n", (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart);

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);

    adjustAABBs<<<params.numParticles/128, 128>>>(params.numParticles, params.particles, aabbBuffer);
    static int numTicks = 0;
    if (++numTicks > 1) {
        numTicks = 0;
        recreateGAS(OPTIX_BUILD_OPERATION_BUILD);
    } else {
        recreateGAS(OPTIX_BUILD_OPERATION_UPDATE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    QueryPerformanceCounter(&t2);
    fprintf(stderr, "rebuild: %lf\n", (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart);
}

static void teardownOptix(void)
{
    LOG_FUNCTION();

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &fbRes, 0));
    CUDA_CHECK(cudaFree((void*)d_param));
    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
    CUDA_CHECK(cudaFree((void*)aabbBuffer));
    CUDA_CHECK(cudaFree((void*)fb));
    CUDA_CHECK(cudaFree((void*)sbt.raygenRecord));
    CUDA_CHECK(cudaFree((void*)sbt.missRecordBase));
    CUDA_CHECK(cudaFree((void*)sbt.hitgroupRecordBase));
    CUDA_CHECK(cudaFree((void*)d_gas_output_buffer));
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));
}

int main()
{
    setupOptix();
    generateParticles();
    recreateGAS(OPTIX_BUILD_OPERATION_BUILD);
    createModule();
    createProgramGroups();
    createPipeline();
    createSBT();
    setupGUI();

    CUDA_CHECK(cudaGraphicsMapResources(1, &fbRes, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, fbRes, 0, 0));

    CUDA_CHECK(cudaMalloc((void**)&d_param, sizeof(struct Params)));

    int simulate = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            simulate = 1;
        }

        if (simulate) {
        
            launch();
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, fbID);
        glBegin(GL_QUADS);
        glTexCoord2i(0, 1); glVertex2i(-1,  1);
        glTexCoord2i(1, 1); glVertex2i( 1,  1);
        glTexCoord2i(1, 0); glVertex2i( 1, -1);
        glTexCoord2i(0, 0); glVertex2i(-1, -1);
        glEnd();

        glfwSwapBuffers(window);
    }
    teardownOptix();
}
