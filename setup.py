from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension(
            'droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/lietorch/eigen')],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-fPIC',
                    '-std=gnu++17',
                    # '-fopenmp',  # OpenMP가 필요하면 주석 해제
                ],
                'nvcc': [
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '--use_fast_math',
                    '-Xcompiler', '-fPIC',
                    # 필요시: '-Xcompiler', '-fopenmp',
                ],
            },
        ),
    ],
    # ninja 관련 이슈 피하고 싶으면 use_ninja=False
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
)
