# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /apps/cmake/3.21.3/bin/cmake

# The command to remove a file.
RM = /apps/cmake/3.21.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liweichang/program/meld_cuda11_dev/plugin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11

# Utility rule file for C++ApiDocs.

# Include any custom commands dependencies for this target.
include docs-source/CMakeFiles/C++ApiDocs.dir/compiler_depend.make

# Include the progress variables for this target.
include docs-source/CMakeFiles/C++ApiDocs.dir/progress.make

docs-source/CMakeFiles/C++ApiDocs: meld-api-c++/index.html
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating C++ API documentation using Doxygen"

meld-api-c++/index.html:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating C++ API documentation using Doxygen"
	/apps/doxygen/1.8.3.1/bin/doxygen /home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11/DoxyfileC++

C++ApiDocs: docs-source/CMakeFiles/C++ApiDocs
C++ApiDocs: meld-api-c++/index.html
C++ApiDocs: docs-source/CMakeFiles/C++ApiDocs.dir/build.make
.PHONY : C++ApiDocs

# Rule to build all files generated by this target.
docs-source/CMakeFiles/C++ApiDocs.dir/build: C++ApiDocs
.PHONY : docs-source/CMakeFiles/C++ApiDocs.dir/build

docs-source/CMakeFiles/C++ApiDocs.dir/clean:
	cd /home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11/docs-source && $(CMAKE_COMMAND) -P CMakeFiles/C++ApiDocs.dir/cmake_clean.cmake
.PHONY : docs-source/CMakeFiles/C++ApiDocs.dir/clean

docs-source/CMakeFiles/C++ApiDocs.dir/depend:
	cd /home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liweichang/program/meld_cuda11_dev/plugin /home/liweichang/program/meld_cuda11_dev/plugin/docs-source /home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11 /home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11/docs-source /home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11/docs-source/CMakeFiles/C++ApiDocs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : docs-source/CMakeFiles/C++ApiDocs.dir/depend

