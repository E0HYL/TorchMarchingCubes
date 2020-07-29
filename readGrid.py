#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   readGrid.py
@Time    :   2020/07/28 12:12:26
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import struct
import numpy as np
"""
    first 4 bytes: the (integer) sampling resolution, 2^d
    last 4 x 2^d x 2^d x ... bytes: the (single precision) floating point values of the implicit function.
"""
filename = 'basket_grid'


def read_until_sep(fhandle, sep=b'\n'):
    s = b''
    while True:
        b = fhandle.read(1)
        if b == sep:
            break
        else:
            s += b
    return s   


def read_grid(filename=filename):
    with open(filename, 'rb') as f:
        f.read(1) # 'G'
        dim = int(read_until_sep(f))
        print(dim)

        read_until_sep(f)
        grid_dim = np.array(read_until_sep(f).split(b' '), dtype=int)
        assert len(grid_dim) == dim
        print(grid_dim)
        f.read(1) # '\n'
        
        transformation = b''
        t_dim = dim + 1
        t_shape = (t_dim, t_dim)
        while True:
            b = f.read(1)
            transformation += b
            if b == b'\n':
                t_dim -= 1
                if not t_dim:
                    break
        transformation = np.fromstring(transformation, dtype=float, sep=' ').reshape(t_shape)
        # print(eval(transformation))
        print(transformation) # gridToModel = voxelToModel = unitCubeToModel * voxelToUnitCube

        grids = f.read()
        grids = np.frombuffer(grids, '<f4')
        assert len(grids) == np.prod(grid_dim)
        grids = grids.reshape(grid_dim)
        print(grids.shape, grids)

    return grids, grid_dim, transformation


if __name__ == "__main__":
    read_grid()