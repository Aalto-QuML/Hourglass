from setuptools import setup, Extension
from torch.utils import cpp_extension

# torch_library_paths = cpp_extension.library_paths(cuda=False)
torch_library_paths = cpp_extension.library_paths()

setup(name='forward_backward_mt',
      ext_modules=[cpp_extension.CppExtension('forward_backward_mt', ['ph/forward_backward_mt_cpu.cpp'], extra_link_args=[
                    '-Wl,-rpath,' + library_path
                    for library_path in torch_library_paths]),],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='forward_backward_mt',
      ext_modules=[cpp_extension.CppExtension('forward_backward_mt', ['ph/forward_backward_mt_cpu.cpp'],# extra_compile_args=['-g', '-O0', '-fno-omit-frame-pointer'],
      extra_link_args=[
                    '-Wl,-rpath,' + library_path
                    for library_path in torch_library_paths]),],
      cmdclass={'build_ext': cpp_extension.BuildExtension})












# from setuptools import setup
# from torch.utils.cpp_extension import CppExtension, BuildExtension

# setup(
#     name='forward_backward_mt',
#     ext_modules=[
#         CppExtension(
#             'forward_backward_mt',
#             ['ph/forward_backward_mt_cpu.cpp'],
#             extra_compile_args={'cxx': ['-O0', '-g', '-fno-omit-frame-pointer', '-fsanitize=address', '-std=c++17']},
#             extra_link_args=['-fsanitize=address'],
#         )
#     ],
#     cmdclass={'build_ext': BuildExtension},
# )
