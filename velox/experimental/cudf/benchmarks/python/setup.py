#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import subprocess

# Get paths from environment or use defaults
VELOX_ROOT = os.environ.get('VELOX_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
BUILD_DIR = os.environ.get('VELOX_BUILD_DIR', os.path.join(VELOX_ROOT, '_build', 'release'))

def get_include_dirs():
    """Get all required include directories."""
    include_dirs = [
        VELOX_ROOT,
        os.path.join(VELOX_ROOT, 'velox'),
        os.path.join(os.path.dirname(__file__), 'cpp', 'tpch'),  # For the TPC-H wrapper header
        os.path.join(os.path.dirname(__file__), 'bindings', 'tpch'),  # For Cython .pxd files
    ]
    
    # Add third-party dependencies from build directory
    deps_dir = os.path.join(BUILD_DIR, '_deps')
    if os.path.exists(deps_dir):
        # List of third-party libraries with their standard include paths
        third_party_includes = [
            # CCCL/Thrust/CUB
            ('cccl-src', 'thrust'),
            ('cccl-src', 'libcudacxx', 'include'),
            ('cccl-src', 'cub'),
            # Common libraries
            ('fmt-src', 'include'),
            ('folly-src'),
            ('gflags-src', 'include'),
            ('glog-src', 'src'),
            ('glog-build'),
            ('double-conversion-src'),
            ('re2-src'),
            ('simdjson-src', 'include'),
            ('spdlog-src', 'include'),
        ]
        
        for path_components in third_party_includes:
            if isinstance(path_components, str):
                path_components = (path_components,)
            
            include_path = os.path.join(deps_dir, *path_components)
            if os.path.exists(include_path):
                include_dirs.append(include_path)
    
    return include_dirs

def get_library_dirs():
    """Get all required library directories."""
    library_dirs = [
        BUILD_DIR,
        # Python bridge library
        os.path.join(BUILD_DIR, 'velox/experimental/cudf/benchmarks/python/cpp/tpch'),
        # CUDF-related libraries
        os.path.join(BUILD_DIR, 'velox/experimental/cudf/benchmarks'),
        os.path.join(BUILD_DIR, 'velox/experimental/cudf/exec'),
        os.path.join(BUILD_DIR, 'velox/experimental/cudf/tests/utils'),
        # TPC-H benchmarks
        os.path.join(BUILD_DIR, 'velox/benchmarks/tpch'),
        # Third-party shared libraries
        os.path.join(BUILD_DIR, '_deps', 'cudf-build'),
        os.path.join(BUILD_DIR, '_deps', 'rapids_logger-build'),
        os.path.join(BUILD_DIR, '_deps', 'curl-build', 'lib'),
    ]
    return [d for d in library_dirs if os.path.exists(d)]

def get_libraries():
    """Get all required libraries to link."""
    # The bridge library should contain all necessary Velox code
    # We only need to link shared libraries that aren't already in the bridge
    return [
        'python_benchmark_bridge_tpch',  # Our TPC-H bridge library (shared)
        'cudf',  # cuDF shared library
        'rapids_logger',  # Rapids logger shared library
        'curl',  # CURL shared library
        'pthread',
        'dl',
        'rt',
    ]

def get_extra_compile_args():
    """Get extra compiler arguments."""
    return [
        '-std=c++17',
        '-O3',
        '-DNDEBUG',
        '-fPIC',
        '-Wno-deprecated-declarations',
    ]

def get_extra_link_args():
    """Get extra linker arguments."""
    args = [
        '-Wl,-rpath,$ORIGIN',
    ]
    
    # Add library paths to rpath
    for lib_dir in get_library_dirs():
        args.append(f'-Wl,-rpath,{lib_dir}')
    
    return args

# Define the extension
extensions = [
    Extension(
        name="cudf_tpch_benchmark",
        sources=["bindings/tpch/cudf_tpch_benchmark.pyx"],
        include_dirs=get_include_dirs(),
        library_dirs=get_library_dirs(),
        libraries=get_libraries(),
        language="c++",
        extra_compile_args=get_extra_compile_args(),
        extra_link_args=get_extra_link_args(),
    )
]

setup(
    name="cudf_tpch_benchmark",
    version="0.1.0",
    description="Python bindings for Velox CUDF TPC-H Benchmarks",
    author="Facebook, Inc.",
    license="Apache 2.0",
    packages=[],  # No Python packages, only C extension module
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'embedsignature': True,
        },
        include_path=[os.path.join(os.path.dirname(__file__), 'bindings', 'tpch')]
    ),
    python_requires='>=3.7',
    install_requires=[
        'cython>=0.29',
    ],
    extras_require={
        'test': [
            'pytest>=6.0',
            'pytest-cov>=2.12',
            'pytest-xdist>=2.3',
        ],
    },
    zip_safe=False,
)
