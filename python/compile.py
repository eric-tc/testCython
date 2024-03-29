from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



ext_modules = [
    Extension("libtrain",  ["libtrain.py"]),
    Extension("libmodel",  ["libmodel.py"]),
    Extension("libcustomDataset",  ["libcustomDataset.py"]),
    Extension("libinference",  ["libinference.py"]),
   
]

setup(
    name = 'My Program Name',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)