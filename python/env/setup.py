import  os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy

# os.environ["CC"] = "clang"

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


ext = Extension('_SPOG',
                sources=['src/SPOG_wrap.cpp',
                		 'src/spog.cpp',
                		 ],
                swig_opts=['-c++'],
                library_dirs=[CUDA['lib64']],
                libraries=['cudart','ntl','gmp', 'spog', 'curand', 'cufft'],
                language='c++',
                runtime_library_dirs=[CUDA['lib64']],
                define_macros = [('NPY_NO_DEPRECATED_API',)],
                extra_compile_args=['-std=c++11', '-g'],
                include_dirs = [numpy_include, CUDA['include'], 'src',"../include", '/usr/local/include/nlohmann'])

# check for swig
if find_in_path('swig', os.environ['PATH']):
    subprocess.check_call('swig -python -c++ -o src/SPOG_wrap.cpp src/spog.i', shell=True)
else:
    raise EnvironmentError('the swig executable was not found in your PATH')

setup(name='SPOG',
      # random metadata. there's more you can supploy
      author='Pedro Alves',
      version='1',


      # this is necessary so that the swigged python file gets picked up
      py_modules=['SPOG'],
      package_dir={'': 'src'},

      ext_modules = [ext],

      zip_safe=False)
