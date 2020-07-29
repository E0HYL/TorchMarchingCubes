#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mcubes.py
@Time    :   2020/07/25 20:46:02
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib

import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
import mcubes_module as mc


def main(gridfile, isovalue):    
    from readGrid import read_grid
    u = read_grid(gridfile)[0]

    u = torch.from_numpy(-u.copy())
    isovalue = -isovalue
    verts, faces = marching_cubes(u, isovalue)

    verts = verts.numpy()
    faces = faces.numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    o3d.io.write_triangle_mesh(f'mc_{gridfile}_{isovalue}.ply', mesh)


def test_mc(isovalue):
    # Grid data
    N = 128
    x, y, z = np.mgrid[:N, :N, :N]
    x = (x / N).astype('float32')
    y = (y / N).astype('float32')
    z = (z / N).astype('float32')

    # Implicit function (metaball)
    f0 = (x - 0.35) ** 2 + (y - 0.35) ** 2 + (z - 0.35) ** 2
    f1 = (x - 0.65) ** 2 + (y - 0.65) ** 2 + (z - 0.65) ** 2
    u = 1.0 / f0 + 1.0 / f1
    rgb = np.stack((x, y, z), axis=-1)
    rgb = np.transpose(rgb, axes=(3, 2, 1, 0)).copy()

    # Test (CPU)
    u = torch.from_numpy(u)
    rgb = torch.from_numpy(rgb)
    verts, faces = marching_cubes(u, isovalue)

    verts = verts.numpy()
    faces = faces.numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    # o3d.visualization.draw_geometries([mesh, wire], window_name='Marching cubes (CPU)')
    o3d.io.write_triangle_mesh(f'mc_{isovalue}.ply', mesh)


def point_with_normals(npfile='./basket_0.01_8.ply'):
    point_cloud = o3d.io.read_point_cloud()
    points = np.asarray(point_cloud.points)
    print(f'Points:\n{points}')
    rgb = np.asarray(point_cloud.colors)
    print(f'Point Colors:\n{rgb}')
    normals = np.asarray(point_cloud.normals)
    print(f'Point Normals:\n{normals}')
    return points, normals, rgb


def marching_cubes(vol, thresh):
    """
    vol: 3D torch tensor
    thresh: threshold
    """
    return mc.mcubes_cpu(vol, thresh)


if __name__ == '__main__':
    # Modules needed for testing
    import numpy as np
    import open3d as o3d
    import argparse
    parser = argparse.ArgumentParser(description='My Marching Cubes')
    parser.add_argument('--test', default=False, help='test only')
    parser.add_argument('--file', default='basket_grid', help='Grid output of possion reconstruction.')
    args = parser.parse_args()

    if args.test:
        test_mc(15)
        test_mc(14)
    else:
        gridfile = args.file
        main(gridfile, 10.011914e-01)
        # main(gridfile, 5.111914e-01)
