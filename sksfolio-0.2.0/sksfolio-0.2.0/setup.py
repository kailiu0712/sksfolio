"""Build the optional native kernels shipped with sksfolio."""

from __future__ import annotations

import os

from setuptools import Extension, setup


compile_args = ["/O2"] if os.name == "nt" else ["-O3", "-std=c11"]

setup(
    ext_modules=[
        Extension(
            "sksfolio._native_pava",
            sources=["sksfolio/_native_pava.c"],
            extra_compile_args=compile_args,
            optional=True,
        )
    ]
)
