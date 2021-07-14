#ifndef OPENMM_CUDAMELDKERNELSOURCES_H_
#define OPENMM_CUDAMELDKERNELSOURCES_H_

#include <string>

namespace MeldPlugin {

/**
 * This class is a central holding place for the source code of CUDA kernels.
 * The CMake build script inserts declarations into it based on the .cu files in the
 * kernels subfolder.
 */

class CudaMeldKernelSources {
public:
static const std::string computeMeld;
static const std::string vectorOps;

};

} // namespace MeldPlugin

#endif /*OPENMM_CUDAMELDKERNELSOURCES_H_*/
