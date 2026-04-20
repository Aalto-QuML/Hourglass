from setuptools import setup, Extension
from torch.utils import cpp_extension

# torch_library_paths = cpp_extension.library_paths(cuda=False)
torch_library_paths = cpp_extension.library_paths()

setup(name='backward_only_mt',
      ext_modules=[cpp_extension.CppExtension('backward_only_mt', ['ph/backward_only_mt_cpu.cpp'], extra_link_args=[
                    '-Wl,-rpath,' + library_path
                    for library_path in torch_library_paths]),],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='backward_only_mt',
      ext_modules=[cpp_extension.CppExtension('backward_only_mt', ['ph/backward_only_mt_cpu.cpp'],# extra_compile_args=['-g', '-O0', '-fno-omit-frame-pointer'],
      extra_link_args=[
                    '-Wl,-rpath,' + library_path
                    for library_path in torch_library_paths]),],
      cmdclass={'build_ext': cpp_extension.BuildExtension})


