#include "rtfluid.h"
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

#define ALIGN_SIZE(x) (((size_t)(x) + 7) & ~7u)

#define LOG_FUNCTION() printf("[%s:%d]: Enter\n", __FILE__, __LINE__)

#define CUDA_CHECK(func) cudaCheckImpl((func), __FILE__, __LINE__)
static void cudaCheckImpl(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
    fprintf(stderr, "[%s:%d]: OK\n", file, line);
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

struct Params params;
extern unsigned char rtfluid_device[];
extern uint32_t rtfluid_deviceLength;
char logBuffer[2048];
size_t logSize = 2048;
OptixDeviceContext context;
OptixAabb *aabb;

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    fprintf(stderr, "[%d][%s]: %s\n", level, tag, message);
}

static void setupOptix(void)
{
    LOG_FUNCTION();

    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(optixInit());

    OptixDeviceContextOptions options = {0};
    memset(&options, 0, sizeof(options));
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    CUDA_CHECK(optixDeviceContextCreate(0 /*=> take current context */, &options, &context));
}

static void generateParticles(void)
{
    params.numParticles = 1000;

    struct Particle *particles = malloc(sizeof(struct Particle)*params.numParticles);
    for (int i = 0; i < params.numParticles; i++) {
        particles[i].x = (2.0*(float)rand()/(float)RAND_MAX)-1.0f;
        particles[i].y = (2.0*(float)rand()/(float)RAND_MAX)-1.0f;
        particles[i].z = (2.0*(float)rand()/(float)RAND_MAX)-1.0f+2.0f;
    }
    
    aabb = malloc(ALIGN_SIZE(params.numParticles * sizeof(OptixAabb)));
    for (int i = 0; i < params.numParticles; i++) {
        aabb[i].minX = particles[i].x-PARTICLE_RADIUS;
        aabb[i].minY = particles[i].y-PARTICLE_RADIUS;
        aabb[i].minZ = particles[i].z-PARTICLE_RADIUS;
        aabb[i].maxX = particles[i].x+PARTICLE_RADIUS;
        aabb[i].maxY = particles[i].y+PARTICLE_RADIUS;
        aabb[i].maxZ = particles[i].z+PARTICLE_RADIUS;
    }

    cudaMalloc((void**)&params.particles, ALIGN_SIZE(sizeof(struct Particle)*params.numParticles));
    cudaMemcpy(params.particles, particles,
        sizeof(struct Particle)*params.numParticles, cudaMemcpyHostToDevice);
    free(particles);
}

OptixTraversableHandle gas_handle;
CUdeviceptr            d_gas_output_buffer;


static void createGAS(void)
{
    LOG_FUNCTION();

    OptixAccelBuildOptions accel_options;
    memset(&accel_options, 0, sizeof(accel_options));
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    CUdeviceptr d_aabb_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_aabb_buffer, ALIGN_SIZE(params.numParticles*sizeof(OptixAabb))));
    CUDA_CHECK(cudaMemcpy((void*)d_aabb_buffer, aabb, params.numParticles*sizeof(OptixAabb), cudaMemcpyHostToDevice));

    OptixBuildInput aabb_input;
    memset(&aabb_input, 0, sizeof(aabb_input));
    aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
    aabb_input.customPrimitiveArray.numPrimitives = params.numParticles;

    uint32_t aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    CUDA_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &aabb_input, 1, &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = ALIGN_SIZE(gas_buffer_sizes.outputSizeInBytes);
    CUDA_CHECK(cudaMalloc((void**)&d_buffer_temp_output_gas_and_compacted_size, compactedSizeOffset));

    OptixAccelEmitDesc emitProperty;
    memset(&emitProperty, 0, sizeof(emitProperty));
    emitProperty.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    CUDA_CHECK(optixAccelBuild(context,
                    0,                  // CUDA stream
                    &accel_options,
                    &aabb_input,
                    1,                  // num build inputs
                    d_temp_buffer_gas,
                    gas_buffer_sizes.tempSizeInBytes,
                    d_buffer_temp_output_gas_and_compacted_size,
                    gas_buffer_sizes.outputSizeInBytes,
                    &gas_handle,
                    &emitProperty,      // emitted property list
                    1                   // num emitted properties
                    ));

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
    CUDA_CHECK(cudaFree((void*)d_aabb_buffer));
}

OptixModule module;
OptixPipelineCompileOptions pipeline_compile_options;

static void createModule(void)
{
    LOG_FUNCTION();

    OptixModuleCompileOptions module_compile_options;
    memset(&module_compile_options, 0, sizeof(module_compile_options));
#if 1
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    
    memset(&pipeline_compile_options, 0, sizeof(OptixPipelineCompileOptions));
    pipeline_compile_options.usesMotionBlur        = 0;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_compile_options.numPayloadValues      = 3;
    pipeline_compile_options.numAttributeValues    = 4; // XXX
    pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    CUDA_CHECK(optixModuleCreate(context, &module_compile_options, &pipeline_compile_options,
                      rtfluid_device, rtfluid_deviceLength, logBuffer, &logSize, &module));
}

OptixProgramGroup raygen_prog_group;
OptixProgramGroup miss_prog_group;
OptixProgramGroup hitgroup_prog_group;

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
    CUDA_CHECK(optixProgramGroupCreate(context, &raygen_prog_group_desc, 1, &program_group_options,
                            logBuffer, &logSize, &raygen_prog_group));
    
    LOG_FUNCTION();

    OptixProgramGroupDesc miss_prog_group_desc;
    memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    CUDA_CHECK(optixProgramGroupCreate(context, &miss_prog_group_desc, 1, &program_group_options,
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

    CUDA_CHECK(optixProgramGroupCreate(context, &hitgroup_prog_group_desc, 1, &program_group_options,
                            logBuffer, &logSize, &hitgroup_prog_group));
}

OptixPipeline pipeline;

static void createPipeline(void)
{
    LOG_FUNCTION();

    const uint32_t max_trace_depth  = 2;
    OptixProgramGroup program_groups[] = {
        raygen_prog_group,
        miss_prog_group,
        hitgroup_prog_group
    };
    const size_t numProgramGroups = 3;

    OptixPipelineLinkOptions pipeline_link_options;
    memset(&pipeline_link_options, 0, sizeof(OptixPipelineLinkOptions));
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    CUDA_CHECK(optixPipelineCreate(context, &pipeline_compile_options, &pipeline_link_options,
                        program_groups, numProgramGroups, logBuffer, &logSize, &pipeline));

    OptixStackSizes stack_sizes;
    memset(&stack_sizes, 0, sizeof(OptixStackSizes));
    for (size_t i = 0; i < numProgramGroups; i++)
        CUDA_CHECK(optixUtilAccumulateStackSizes(program_groups[i], &stack_sizes, pipeline));

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    CUDA_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0, &direct_callable_stack_size_from_traversal,
                               &direct_callable_stack_size_from_state, &continuation_stack_size));
    CUDA_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                              direct_callable_stack_size_from_state, continuation_stack_size, 1));
}

OptixShaderBindingTable sbt;

static void createSBT(void)
{
    LOG_FUNCTION();

    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(struct RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc((void**)&raygen_record, raygen_record_size));

    struct RayGenSbtRecord rg_sbt;
    CUDA_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy((void*)raygen_record, &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));


    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof(struct MissSbtRecord);
    CUDA_CHECK(cudaMalloc((void**)&miss_record, miss_record_size));
    
    struct MissSbtRecord ms_sbt;
    CUDA_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy((void*)miss_record, &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));


    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof(struct HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc((void**)&hitgroup_record, hitgroup_record_size));
    
    struct HitGroupSbtRecord hg_sbt;
    CUDA_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy((void*)hitgroup_record, &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));


    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof(struct MissSbtRecord);
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(struct HitGroupSbtRecord);
    sbt.hitgroupRecordCount         = 1;
}

const int width  = 800;
const int height = 600;
const int comp   = 4;

GLFWwindow *window;
GLuint fbID;
uint8_t *fb;
cudaGraphicsResource_t fbRes;
cudaArray_t array;

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

static void launch(void)
{
    LOG_FUNCTION();

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    params.image        = fb;
    params.image_width  = width;
    params.image_height = height;
    params.gas_handle   = gas_handle;

    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc((void**)&d_param, sizeof(struct Params)));
    CUDA_CHECK(cudaMemcpy((void*)d_param, &params, sizeof(params), cudaMemcpyHostToDevice));

    CUDA_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(struct Params), &sbt,
                params.image_width, params.image_height, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Transfer from CUDA to OpenGL
    cudaGraphicsMapResources(1, &fbRes, 0);
    cudaGraphicsSubResourceGetMappedArray(&array, fbRes, 0, 0);
    cudaMemcpy2DToArray(array, 0, 0, fb, width * sizeof(uint32_t),
        width * sizeof(uint32_t), height, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &fbRes, 0);

    CUDA_CHECK(cudaFree((void*)d_param));
}

static void teardownOptix(void)
{
    LOG_FUNCTION();

    CUDA_CHECK(cudaFree((void*)fb));
    CUDA_CHECK(cudaFree((void*)sbt.raygenRecord));
    CUDA_CHECK(cudaFree((void*)sbt.missRecordBase));
    CUDA_CHECK(cudaFree((void*)sbt.hitgroupRecordBase));
    CUDA_CHECK(cudaFree((void*)d_gas_output_buffer));
    CUDA_CHECK(optixPipelineDestroy(pipeline));
    CUDA_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    CUDA_CHECK(optixProgramGroupDestroy(miss_prog_group));
    CUDA_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    CUDA_CHECK(optixModuleDestroy(module));
    CUDA_CHECK(optixDeviceContextDestroy(context));
}

int main()
{
    setupOptix();
    generateParticles();
    createGAS();
    createModule();
    createProgramGroups();
    createPipeline();
    createSBT();
    
    setupGUI();
    launch();
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        Sleep(10);

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
