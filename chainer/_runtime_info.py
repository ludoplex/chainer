import sys

import numpy
import six

import chainer
from chainer.backends import cuda


class _RuntimeInfo(object):

    chainer_version = None
    numpy_version = None
    cuda_info = None

    def __init__(self):
        self.chainer_version = chainer.__version__
        self.numpy_version = numpy.__version__
        self.cuda_info = cuda.cupyx.get_runtime_info() if cuda.available else None

    def __str__(self):
        s = six.StringIO()
        s.write(f'''Chainer: {self.chainer_version}\n''')
        s.write(f'''NumPy: {self.numpy_version}\n''')
        if self.cuda_info is None:
            s.write('''CuPy: Not Available\n''')
        else:
            s.write('''CuPy:\n''')
            for line in str(self.cuda_info).splitlines():
                s.write(f'''  {line}\n''')
        return s.getvalue()


def get_runtime_info():
    return _RuntimeInfo()


def print_runtime_info(out=None):
    if out is None:
        out = sys.stdout
    out.write(str(get_runtime_info()))
    if hasattr(out, 'flush'):
        out.flush()
