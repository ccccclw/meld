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
CMAKE_BINARY_DIR = /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0

# Include any dependencies generated for this target.
include CMakeFiles/MeldPlugin.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MeldPlugin.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MeldPlugin.dir/flags.make

CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.o: CMakeFiles/MeldPlugin.dir/flags.make
CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.o: ../openmmapi/src/MeldForce.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.o"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.o -c /home/liweichang/program/meld_cuda10_dev/plugin/openmmapi/src/MeldForce.cpp

CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.i"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweichang/program/meld_cuda10_dev/plugin/openmmapi/src/MeldForce.cpp > CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.i

CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.s"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweichang/program/meld_cuda10_dev/plugin/openmmapi/src/MeldForce.cpp -o CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.s

CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.o: CMakeFiles/MeldPlugin.dir/flags.make
CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.o: ../openmmapi/src/MeldForceImpl.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.o"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.o -c /home/liweichang/program/meld_cuda10_dev/plugin/openmmapi/src/MeldForceImpl.cpp

CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.i"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweichang/program/meld_cuda10_dev/plugin/openmmapi/src/MeldForceImpl.cpp > CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.i

CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.s"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweichang/program/meld_cuda10_dev/plugin/openmmapi/src/MeldForceImpl.cpp -o CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.s

CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.o: CMakeFiles/MeldPlugin.dir/flags.make
CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.o: ../serialization/src/MeldForceProxy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.o"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.o -c /home/liweichang/program/meld_cuda10_dev/plugin/serialization/src/MeldForceProxy.cpp

CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.i"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweichang/program/meld_cuda10_dev/plugin/serialization/src/MeldForceProxy.cpp > CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.i

CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.s"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweichang/program/meld_cuda10_dev/plugin/serialization/src/MeldForceProxy.cpp -o CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.s

CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.o: CMakeFiles/MeldPlugin.dir/flags.make
CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.o: ../serialization/src/MeldSerializationProxyRegistration.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.o"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.o -c /home/liweichang/program/meld_cuda10_dev/plugin/serialization/src/MeldSerializationProxyRegistration.cpp

CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.i"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweichang/program/meld_cuda10_dev/plugin/serialization/src/MeldSerializationProxyRegistration.cpp > CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.i

CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.s"
	/apps/mpi/cuda/10.0.130/gcc/9.3.0/openmpi/3.1.2/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweichang/program/meld_cuda10_dev/plugin/serialization/src/MeldSerializationProxyRegistration.cpp -o CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.s

# Object files for target MeldPlugin
MeldPlugin_OBJECTS = \
"CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.o" \
"CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.o" \
"CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.o" \
"CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.o"

# External object files for target MeldPlugin
MeldPlugin_EXTERNAL_OBJECTS =

libMeldPlugin.so: CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForce.cpp.o
libMeldPlugin.so: CMakeFiles/MeldPlugin.dir/openmmapi/src/MeldForceImpl.cpp.o
libMeldPlugin.so: CMakeFiles/MeldPlugin.dir/serialization/src/MeldForceProxy.cpp.o
libMeldPlugin.so: CMakeFiles/MeldPlugin.dir/serialization/src/MeldSerializationProxyRegistration.cpp.o
libMeldPlugin.so: CMakeFiles/MeldPlugin.dir/build.make
libMeldPlugin.so: CMakeFiles/MeldPlugin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared library libMeldPlugin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MeldPlugin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MeldPlugin.dir/build: libMeldPlugin.so

.PHONY : CMakeFiles/MeldPlugin.dir/build

CMakeFiles/MeldPlugin.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MeldPlugin.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MeldPlugin.dir/clean

CMakeFiles/MeldPlugin.dir/depend:
	cd /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liweichang/program/meld_cuda10_dev/plugin /home/liweichang/program/meld_cuda10_dev/plugin /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0 /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0 /home/liweichang/program/meld_cuda10_dev/plugin/build_cuda10_dev_0/CMakeFiles/MeldPlugin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MeldPlugin.dir/depend

