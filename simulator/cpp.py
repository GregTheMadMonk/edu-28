import os

CPP_BASE_PATH = f"{os.path.dirname(__file__)}/cpp"

from torch.utils.cpp_extension import load

__cppmod = None

def get(realType = None):
    """!
    \brief Get C++ extension interface

    \param realType Module real type

    \throws RuntimeError if called with arguments after the extension has been initialized

    On the first call, compiles and loads the C++ extension module
    """

    global __cppmod
    CPP_REAL = "double"
    if realType is not None:
        if __cppmod is not None:
            raise RuntimeError("Compile-time flags provided for an already loaded C++ extension")

        print(f"Custom simulator C++ real type set to {realType}")

    if __cppmod is None:
        print(f"Loading C++ submodule from {CPP_BASE_PATH}")
        __cppmod = load(
            name = "cpp",
            build_directory = CPP_BASE_PATH,
            sources = f"{CPP_BASE_PATH}/extension.cc",
            extra_cflags = [ f"-DREAL={realType} -O3 -std=c++20 -DNDEBUG" ],
            verbose = False
        )

    return __cppmod
