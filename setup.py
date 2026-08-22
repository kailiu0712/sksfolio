"""Build the optional partial-sort PAVA kernel shipped with sksfolio."""

from __future__ import annotations

import os

from setuptools import Extension, setup


compile_args = ["/O2"] if os.name == "nt" else ["-O3", "-std=c11"]

native_sources = {
    "sksfolio._native_pava": "sksfolio/_native_pava.c",
}


setup(
    ext_modules=[
        Extension(
            module,
            sources=[source],
            extra_compile_args=compile_args,
            optional=True,
        )
        for module, source in native_sources.items()
    ]
)
