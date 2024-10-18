from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        name='asynchronous_graph_spi_cpu',
        sources=[
            "/home/ff/code/spike_new/cpp_files_cpu/Util.cpp",
            "/home/ff/code/spike_new/cpp_files_cpu/AsynchronousGraphSpiNet.cpp",
            "/home/ff/code/spike_new/cpp_files_cpu/AsynchronousConvLayer.cpp",
            "/home/ff/code/spike_new/cpp_files_cpu/SpikeGen.cpp",
        ],
        include_dirs=[
            "/home/ff/code/spike_new/cpp_files_cpu",
            "/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/pybind11/include",  # 示例路径，根据实际安装调整
            "/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/torch/include",  # 示例路径，根据实际安装调整
            "/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/torch/include/torch/csrc/api/include",  # 示例路径
            "/usr/local/cuda/include",  # 示例CUDA安装路径
        ],
        library_dirs=[
            "/home/ff/anaconda3/envs/spike/lib/python3.9/site-packages/torch/lib",
            "/usr/local/cuda/lib64",  # 示例CUDA库路径
            "/usr/local/cuda/extras/CUPTI/lib64"
        ],
        libraries=['torch', 'torch_cpu', 'c10', 'cuda', 'cudart', 'cublas', 'cudnn', 'cupti'],
        language='c++',
        extra_compile_args=['-std=c++17'],  # GCC/G++的C++17标准参数
    ),
]

setup(
    name='asynchronous_graph_spi_cpu',
    version='0.1',
    author='fans',
    author_email='fans19990621@gmail.com',  # 修改邮箱地址错误
    description='Parallel processing for Python code',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)