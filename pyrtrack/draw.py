#    Copyright 2009 Mathieu Leocmach
#
#    This file is part of pyrTrack.
#
#    Colloids is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Colloids is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Colloids.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np

def draw_spheres(shape, pos, radii):
    im = np.zeros(shape, bool)
    if np.isscalar(radii):
        radii = radii * np.ones(len(pos))
    assert len(pos)==len(radii)
    assert np.min(pos.max(0)<shape[::-1]) and np.min([0,0,0]<=pos.min(0)), "points out of bounds"
    for p, rsq, m, M in zip(
        pos, radii**2,
        np.maximum(0, pos[:,::-1] - radii[:,None]).astype(int),
        np.minimum(im.shape, pos[:,::-1] + radii[:,None] + 1).astype(int)
    ):
        im[m[0]:M[0], m[1]:M[1], m[2]:M[2]] |= (
            (p[2] - np.arange(m[0], M[0]))[:,None,None]**2 +
            (p[1] - np.arange(m[1], M[1]))[None,:,None]**2 +
            (p[0] - np.arange(m[2], M[2]))[None,None,:]**2 <= rsq
        )
    return im

def draw_rods(pos, radii, shape=None, im=None):
    if im is None:
        im = np.zeros(shape, bool)
    if shape is None:
        shape = im.shape
    assert len(pos)==len(radii)
    assert np.min(pos.max(0)<shape[::-1]) and np.min([0,0,0]<=pos.min(0)), "points out of bounds"
    for p, rsq, m, M in zip(
        pos, radii**2,
        np.maximum(0, pos[:,::-1] - radii[:,None]).astype(int),
        np.minimum(im.shape, pos[:,::-1] + radii[:,None] + 1).astype(int)
    ):
        im[m[0]:M[0], m[1]:M[1], m[2]:M[2]] |= (
            (p[1] - np.arange(m[1], M[1]))[None,:,None]**2 +
            (p[0] - np.arange(m[2], M[2]))[None,None,:]**2 <= rsq
        )
    return im;
