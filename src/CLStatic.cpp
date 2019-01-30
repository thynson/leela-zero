#include "OpenCL.h"
namespace cl {

    std::once_flag Device::default_initialized_;
    Device Device::default_;
    cl_int Device::default_error_ = CL_SUCCESS;

    Device Device::getDefault(cl_int *errResult)
    {
        std::call_once(default_initialized_, makeDefault);
        detail::errHandler(default_error_);
        if (errResult != NULL) {
            *errResult = default_error_;
        }
        return default_;
    }
    Device Device::setDefault(const Device &default_device)
    {
        std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_device));
        detail::errHandler(default_error_);
        return default_;
    }

    std::once_flag Platform::default_initialized_;
    Platform Platform::default_;
    cl_int Platform::default_error_ = CL_SUCCESS;


    void Platform::makeDefaultProvided(const Platform &p) {
       default_ = p;
    }

    Platform Platform::getDefault(
        cl_int *errResult)
    {
        std::call_once(default_initialized_, makeDefault);
        detail::errHandler(default_error_);
        if (errResult != NULL) {
            *errResult = default_error_;
        }
        return default_;
    }
    Platform Platform::setDefault(const Platform &default_platform)
    {
        std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_platform));
        detail::errHandler(default_error_);
        return default_;
    }

    void Platform::makeDefault() {
        /* Throwing an exception from a call_once invocation does not do
        * what we wish, so we catch it and save the error.
        */
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        try
#endif
        {
            // If default wasn't passed ,generate one
            // Otherwise set it
            cl_uint n = 0;

            cl_int err = ::clGetPlatformIDs(0, NULL, &n);
            if (err != CL_SUCCESS) {
                default_error_ = err;
                return;
            }
            if (n == 0) {
                default_error_ = CL_INVALID_PLATFORM;
                return;
            }

            vector<cl_platform_id> ids(n);
            err = ::clGetPlatformIDs(n, ids.data(), NULL);
            if (err != CL_SUCCESS) {
                default_error_ = err;
                return;
            }

            default_ = Platform(ids[0]);
        }
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        catch (cl::Error &e) {
            default_error_ = e.err();
        }
#endif
    }

    std::once_flag Context::default_initialized_;
    Context Context::default_;
    cl_int Context::default_error_ = CL_SUCCESS;

    Context Context::getDefault(cl_int * err)
    {
        std::call_once(default_initialized_, makeDefault);
        detail::errHandler(default_error_);
        if (err != NULL) {
            *err = default_error_;
        }
        return default_;
    }
    Context Context::setDefault(const Context &default_context)
    {
        std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_context));
        detail::errHandler(default_error_);
        return default_;
    }

    std::once_flag CommandQueue::default_initialized_;
    CommandQueue CommandQueue::default_;
    cl_int CommandQueue::default_error_ = CL_SUCCESS;

    CommandQueue CommandQueue::getDefault(cl_int * err)
    {
        std::call_once(default_initialized_, makeDefault);
        detail::errHandler(default_error_, default_create_error);
        if (err != NULL) {
            *err = default_error_;
        }
        return default_;
    }

    CommandQueue CommandQueue::setDefault(const CommandQueue &default_queue)
    {
        std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_queue));
        detail::errHandler(default_error_);
        return default_;
    }
} // namespace cl