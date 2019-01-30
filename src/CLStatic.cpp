#include "OpenCL.h"
namespace cl {

    std::once_flag Device::default_initialized_;
    Device Device::default_;
    cl_int Device::default_error_ = CL_SUCCESS;

    std::once_flag Platform::default_initialized_;
    Platform Platform::default_;
    cl_int Platform::default_error_ = CL_SUCCESS;

    std::once_flag Context::default_initialized_;
    Context Context::default_;
    cl_int Context::default_error_ = CL_SUCCESS;

    std::once_flag CommandQueue::default_initialized_;
    CommandQueue CommandQueue::default_;
    cl_int CommandQueue::default_error_ = CL_SUCCESS;
} // namespace cl