# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /apps/cmake/3.15.6/bin/cmake

# The command to remove a file.
RM = /apps/cmake/3.15.6/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liweichang/program/meld_cuda10_dev/plugin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev

# Include any dependencies generated for this target.
include platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/depend.make

# Include the progress variables for this target.
include platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/progress.make

# Include the compile flags for this target's objects.
include platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/flags.make

platforms/cuda/src/CudaMeldKernelSources.cpp: ../platforms/cuda/src/kernels/computeMeld.cu
platforms/cuda/src/CudaMeldKernelSources.cpp: ../platforms/cuda/src/kernels/vectorOps.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating src/CudaMeldKernelSources.cpp, src/CudaMeldKernelSources.h"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/cmake/3.15.6/bin/cmake -D CUDA_SOURCE_DIR=/home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/src -D CUDA_KERNELS_CPP=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/src/CudaMeldKernelSources.cpp -D CUDA_KERNELS_H=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/src/CudaMeldKernelSources.h -D CUDA_SOURCE_CLASS=CudaMeldKernelSources -P /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/EncodeCUDAFiles.cmake

platforms/cuda/src/CudaMeldKernelSources.h: platforms/cuda/src/CudaMeldKernelSources.cpp
	@$(CMAKE_COMMAND) -E touch_nocreate platforms/cuda/src/CudaMeldKernelSources.h

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.o: platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/flags.make
platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.o: ../platforms/cuda/src/MeldCudaKernelFactory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.o"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.o -c /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/src/MeldCudaKernelFactory.cpp

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.i"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/src/MeldCudaKernelFactory.cpp > CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.i

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.s"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/src/MeldCudaKernelFactory.cpp -o CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.s

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.o: platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/flags.make
platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.o: ../platforms/cuda/src/MeldCudaKernels.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.o"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.o -c /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/src/MeldCudaKernels.cpp

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.i"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/src/MeldCudaKernels.cpp > CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.i

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.s"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/src/MeldCudaKernels.cpp -o CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.s

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.o: platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/flags.make
platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.o: platforms/cuda/src/CudaMeldKernelSources.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.o"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.o -c /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/src/CudaMeldKernelSources.cpp

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.i"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/src/CudaMeldKernelSources.cpp > CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.i

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.s"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/src/CudaMeldKernelSources.cpp -o CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.s

# Object files for target MeldPluginCUDA
MeldPluginCUDA_OBJECTS = \
"CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.o" \
"CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.o" \
"CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.o"

# External object files for target MeldPluginCUDA
MeldPluginCUDA_EXTERNAL_OBJECTS =

platforms/cuda/libMeldPluginCUDA.so: platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernelFactory.cpp.o
platforms/cuda/libMeldPluginCUDA.so: platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/MeldCudaKernels.cpp.o
platforms/cuda/libMeldPluginCUDA.so: platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/src/CudaMeldKernelSources.cpp.o
platforms/cuda/libMeldPluginCUDA.so: platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/build.make
platforms/cuda/libMeldPluginCUDA.so: /apps/compilers/cuda/10.0.130/lib64/libcudart_static.a
platforms/cuda/libMeldPluginCUDA.so: /usr/lib64/librt.so
platforms/cuda/libMeldPluginCUDA.so: libMeldPlugin.so
platforms/cuda/libMeldPluginCUDA.so: platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared library libMeldPluginCUDA.so"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MeldPluginCUDA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/build: platforms/cuda/libMeldPluginCUDA.so

.PHONY : platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/build

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/clean:
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda && $(CMAKE_COMMAND) -P CMakeFiles/MeldPluginCUDA.dir/cmake_clean.cmake
.PHONY : platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/clean

platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/depend: platforms/cuda/src/CudaMeldKernelSources.cpp
platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/depend: platforms/cuda/src/CudaMeldKernelSources.h
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liweichang/program/meld_cuda10_dev/plugin /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : platforms/cuda/CMakeFiles/MeldPluginCUDA.dir/depend

