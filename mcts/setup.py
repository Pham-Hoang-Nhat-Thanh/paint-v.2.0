from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "mcts.mcts_fast",
        ["mcts/mcts_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[
            "-O3",
            "-march=haswell",        # or "core-avx2" if you prefer
            "-fno-fast-math",        # ← ABSOLUTELY REQUIRED
            "-fno-tree-vectorize",   # ← PREVENTS _ZGV*
            "-pthread",
        ],
        libraries=["m"],            # ← ONLY libm
        language="c",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    zip_safe=False,
)
