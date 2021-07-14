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
include platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/depend.make

# Include the progress variables for this target.
include platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/progress.make

# Include the compile flags for this target's objects.
include platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/flags.make

platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.o: platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/flags.make
platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.o: ../platforms/cuda/tests/TestCudaMeldForce.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.o"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/tests && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.o -c /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/tests/TestCudaMeldForce.cpp

platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.i"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/tests && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/tests/TestCudaMeldForce.cpp > CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.i

platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.s"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/tests && /apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/tests/TestCudaMeldForce.cpp -o CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.s

# Object files for target TestCudaMeldForce
TestCudaMeldForce_OBJECTS = \
"CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.o"

# External object files for target TestCudaMeldForce
TestCudaMeldForce_EXTERNAL_OBJECTS =

platforms/cuda/tests/TestCudaMeldForce: platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/TestCudaMeldForce.cpp.o
platforms/cuda/tests/TestCudaMeldForce: platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/build.make
platforms/cuda/tests/TestCudaMeldForce: platforms/cuda/libMeldPluginCUDA.so
platforms/cuda/tests/TestCudaMeldForce: libMeldPlugin.so
platforms/cuda/tests/TestCudaMeldForce: /apps/compilers/cuda/10.0.130/lib64/libcudart_static.a
platforms/cuda/tests/TestCudaMeldForce: /usr/lib64/librt.so
platforms/cuda/tests/TestCudaMeldForce: platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable TestCudaMeldForce"
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TestCudaMeldForce.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/build: platforms/cuda/tests/TestCudaMeldForce

.PHONY : platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/build

platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/clean:
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/tests && $(CMAKE_COMMAND) -P CMakeFiles/TestCudaMeldForce.dir/cmake_clean.cmake
.PHONY : platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/clean

platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/depend:
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liweichang/program/meld_cuda10_dev/plugin /home/liweichang/program/meld_cuda10_dev/plugin/platforms/cuda/tests /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/tests /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev/platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : platforms/cuda/tests/CMakeFiles/TestCudaMeldForce.dir/depend

