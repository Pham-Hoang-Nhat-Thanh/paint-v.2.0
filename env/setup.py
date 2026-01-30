from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "env.network_fast",
        ["env/network_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[
            "-O3",
            "-march=native", 
            "-ffast-math",
            "-std=c++11"  # For C++ STL containers if needed
        ],
        language="c++",  # Using C++ for STL vectors if needed, or change to "c"
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ),
    zip_safe=False,
)