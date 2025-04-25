# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
#所以这里是有  ..lib的

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)#获取当前文件路径

lib_path = osp.join(this_dir, '..', 'lib') #构建新的路径lib_path
add_path(lib_path)  #调用add_path将这个路径添加到sys.path中
