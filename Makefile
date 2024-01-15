.PHONY : all clean
all : build/rtfluid

GLFW_INCLUDE := "C:\glfw\include"
GLFW_LIB := "C:\glfw\lib-vc2022"
OPTIX_INCLUDE := "C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0\include"
CUDA_LIB := "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64"

NVCC := "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe"
BIN2C := "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/bin2c.exe"

CC := "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64/cl.exe"

build/rtfluid_device.ptx : src/rtfluid_device.cu | build
	$(NVCC) -dlto -dlink -ptx -ccbin $(CC) \
		--compiler-options "/std:c11" -o build/rtfluid_device.ptx \
		-I $(OPTIX_INCLUDE) src/rtfluid_device.cu

build/rtfluid_device.c : build/rtfluid_device.ptx | build
	printf '%s\n%s\n' "#include <stdint.h>" \
		"$$($(BIN2C) --name rtfluid_device --length --stdint build/rtfluid_device.ptx)" \
		>build/rtfluid_device.c

build/rtfluid : src/rtfluid.c build/rtfluid_device.c | build
	$(NVCC) -ccbin $(CC) --compiler-options "/std:c11" -o build/rtfluid \
		-I $(OPTIX_INCLUDE) -I $(GLFW_INCLUDE) src/rtfluid.c build/rtfluid_device.c \
		-L $(CUDA_LIB) -L $(GLFW_LIB) -lcuda -lcudart -lnvrtc -lAdvapi32 -lglfw3_mt \
		-lOpenGL32 -lGDI32 -lKernel32 -lUser32 -lShell32

clean : build/rtfluid
	-rm -f build/rtfluid

build :
	mkdir build