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
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.spatial import KDTree
from numba import jit, vectorize
import math

def get_bonds(positions, radii, maxdist=3.0):
    """Bonds by relative distances, such that $r_{ij} < maxdist (R_i + R_j)$.

    Returns pairs, distances. Pairs are sorted and unique."""
    assert len(positions)==len(radii)
    tree = KDTree(positions)
    rmax = radii.max()
    #fetch all potential pairs, already sorted
    pairs = tree.query_pairs(2*rmax*maxdist, output_type='ndarray')
    if len(pairs) == 0:
        return np.zeros((0,2), int), np.zeros(0)
    #compute all pair's square distances via numpy
    dists = np.sum((positions[pairs[:,0]] - positions[pairs[:,1]])**2, -1)
    #filter out the pairs that are too far
    good = dists < maxdist**2 * radii[pairs].sum(-1)**2
    return pairs[good], np.sqrt(dists[good])

def radius2scale(R, k=1.6, n=3.0, dim=3):
    """Converts a radius (in pixels) to a scale index (logarithmic scale corresponding to the inner working of a MultiscaleTracker)"""
    return n / np.log(2) * np.log(
        R/k * np.sqrt(n*(2**(2.0/n) - 1)/(2 * dim* np.log(2)))
        ) -1

def scale2radius(x, k=1.6, n=3.0, dim=3):
    """Converts a scale index (logarithmic scale corresponding to the inner working of a MultiscaleTracker) to a radius (in pixels)"""
    return k * 2**((x+1)/float(n))*np.sqrt(
        2 * dim * np.log(2) / float(n) / (2**(2.0/n)-1)
        )

def radius2sigma(R, n=3.0, dim=3):
    """Converts a radius (in pixels) to a scale (logarithmic pixel scale)"""
    return R / np.sqrt(2*dim* np.log(2) / n/(1 - 2**(-2.0/n)))

def sigma2radius(sigma, n=3.0, dim=3):
    """Converts a scale (logarithmic pixel scale) to a radius (in pixels)"""
    return sigma * np.sqrt(2*dim* np.log(2) / n/(1 - 2**(-2.0/n)))

@vectorize(["float64(float64, float64, float64)"])
def halfG(d, R, sigma):
    """helper function for G"""
    return math.exp(-(R+d)**2/(2*sigma**2)) * math.sqrt(2/np.pi)*sigma/d + math.erf((R+d)/sigma/math.sqrt(2))

@vectorize(["float64(float64, float64)"])
def midG(R,sigma):
    """helper function for G"""
    x = R/sigma/math.sqrt(2);
    return math.erf(x) - x*math.exp(-x**2)*2/math.sqrt(np.pi)

def G(R, sigma, d=None):
    """Value of the Gaussian blur sigma of a step of radius R (at a distance d)"""
    if d is None:
        return midG(R,sigma)
    return halfG(d,R,sigma) + halfG(-d, R, sigma)

def DoG(R, sigma, alpha, d=None):
    """Value of the difference of Gaussian blurs alpha*sigma and sigma of a step of radius R (at a distance d)"""
    return G(R, alpha*sigma, d) - G(R, sigma, d)

@vectorize(["float64(float64, float64, float64)"])
def halfG_dsigma(d, R, s2):
    """helper function for G_dsigma"""
    return (R**2 + d*R + s2)*math.exp(-(R+d)**2/(2*s2))/math.sqrt(2*np.pi)/d/s2

@vectorize(["float64(float64, float64)"])
def midG_dsigma(R,sigma):
    """helper function for G_dsigma"""
    return -R**3/sigma**4 *math.sqrt(2/np.pi)*math.exp(-R**2/2/sigma**2)

def G_dsigma(R, sigma, d=None):
    """Value of the derivative along R of the Gaussian blur sigma of a step of radius R (at a distance d)"""
    if d is None:
        return midG_dsigma(R,sigma)
    s2 = sigma**2
    return halfG_dsigma(d, R, s2) + halfG_dsigma(-d, R, s2)

def DoG_dsigma(R, sigma, alpha, d=None):
    """Value of the derivative along R of the difference of Gaussian blurs alpha*sigma and sigma of a step of radius R (at a distance d)"""
    return alpha*G_dsigma(R, alpha*sigma, d) - G_dsigma(R, sigma, d)

@vectorize(["float64(float64, float64, float64)"])
def halfG_dsigma_dR(d, R, s2):
    """helper function for G_dsigma_dR"""
    return -R * ((R+d)**2 - s2) * math.exp(-(R+d)**2/(2*s2))/math.sqrt(2*np.pi)/d/(s2**2)

@vectorize(["float64(float64, float64)"])
def midG_dsigma_dR(R,sigma):
    """helper function for G_dsigma_dR"""
    return R**2 * (R**2 - 3*sigma**2)/sigma**6 * math.sqrt(2/np.pi)*math.exp(-R**2/2/sigma**2)

def G_dsigma_dR(R, sigma, d=None):
    """Value of the derivative along R and sigma of the Gaussian blur sigma of a step of radius R (at a distance d)"""
    if d is None:
        return midG_dsigma_dR(R,sigma)
    s2 = sigma**2
    return halfG_dsigma_dR(d, R, s2) + halfG_dsigma_dR(-d, R, s2)

def DoG_dsigma_dR(R, sigma, alpha, d=None):
    """Value of the derivative along R and sigma of the difference of Gaussian blurs alpha*sigma and sigma of a step of radius R (at a distance d)"""
    return alpha*G_dsigma_dR(R, alpha*sigma, d) - G_dsigma_dR(R, sigma, d)

def global_rescale(sigma0, bonds, dists, R0=None, n=3):
    """Takes into account the overlapping of the blurred spot of neighbouring particles to compute the radii of all particles. Suppose all particles equally bright.

    parameters
    ----------
    sigma0 :  array((N))
        The radii output by MultiscaleTracker via scale2radius.
    bonds : array((M,2), int)
        First output of particles.get_bonds
    dists : array((M), float)
        Second output of particles.get_bonds
    R0 : array((N))
        Previous iteration's radii
    n : int
        Same as in MultiscaleTracker
    """
    assert len(bonds)==len(dists)
    alpha = 2**(1.0/n)
    if R0 is None:
        R0 = sigma2radius(sigma0, n=float(n))
    jacob = sparse.lil_matrix(tuple([len(sigma0)]*2))
    v0 = DoG_dsigma(R0, sigma0, alpha)
    for d, (i,j) in zip(np.maximum(dists, R0[bonds].sum(-1)), bonds):
        #d = max(dists[b], R0[i] + R0[j])
        jacob[i,j] = DoG_dsigma_dR(R0[j], sigma0[i], alpha, d)
        jacob[j,i] = DoG_dsigma_dR(R0[i], sigma0[j], alpha, d)
        v0[i] += DoG_dsigma(R0[j], sigma0[i], alpha, d)
        v0[j] += DoG_dsigma(R0[i], sigma0[j], alpha, d)
    for i,a in enumerate(DoG_dsigma_dR(R0, sigma0, alpha)):
        jacob[i,i] = a
    return R0 + spsolve(jacob.tocsc(), -v0)

def global_rescale_intensity(sigma0, bonds, dists, intensities, R0=None, n=3):
    """Takes into account the overlapping of the blurred spot of neighbouring particles to compute the radii of all particles. The brightness of the particles is taken into account.

    parameters
    ----------
    sigma0 :  array((N))
        The radii output by MultiscaleTracker via scale2radius.
    bonds : array((M,2), int)
        First output of particles.get_bonds
    dists : array((M), float)
        Second output of particles.get_bonds
    intensities : array((N), float)
        Output of solve_intensities
    R0 : array((N), float)
        Previous iteration's radii
    n : int
        Same as in MultiscaleTracker
    """
    assert len(bonds)==len(dists)
    alpha = 2**(1.0/n)
    if R0 is None:
        R0 = sigma2radius(sigma0, n=float(n))
    jacob = sparse.lil_matrix(tuple([len(sigma0)]*2))
    v0 = intensities * DoG_dsigma(R0, sigma0, alpha)
    for d, (i,j) in zip(np.maximum(dists, R0[bonds].sum(-1)), bonds):
        #d = max(dists[b], R0[i] + R0[j])
        jacob[i,j] = intensities[j] * DoG_dsigma_dR(R0[j], sigma0[i], alpha, d)
        jacob[j,i] = intensities[i] * DoG_dsigma_dR(R0[i], sigma0[j], alpha, d)
        v0[i] += intensities[j] * DoG_dsigma(R0[j], sigma0[i], alpha, d)
        v0[j] += intensities[i] * DoG_dsigma(R0[i], sigma0[j], alpha, d)
    for i,a in enumerate(intensities * DoG_dsigma_dR(R0, sigma0, alpha)):
        jacob[i,i] = a
    return R0 + spsolve(jacob.tocsc(), -v0)

def solve_intensities(sigma0, bonds, dists, intensities, R0=None, n=3):
    """Takes into account the overlapping of the blurred spot of neighbouring particles to compute the brightness of all particles.

    parameters
    ----------
    sigma0 :  array((N))
        The radii output by MultiscaleTracker via scale2radius.
    bonds : array((M,2), int)
        First output of particles.get_bonds
    dists : array((M), float)
        Second output of particles.get_bonds
    intensities : array((N), float)
        Value of the Difference of Gaussian at the place and scale of each particle, e.g. the last column of the output of MultiscaleTracker
    R0 : array((N), float)
        Previous iteration's radii
    n : int
        Same as in MultiscaleTracker
    """
    assert len(bonds)==len(dists)
    alpha = 2**(1.0/n)
    if R0 is None:
        R0 = sigma2radius(sigma0, n=float(n))
    mat = sparse.lil_matrix(tuple([len(sigma0)]*2))
    for d, (i,j) in zip(dists, bonds):
        mat[i,j] = DoG(R0[j], sigma0[i], alpha, d)
        mat[j,i] = DoG(R0[i], sigma0[j], alpha, d)
    for i,a in enumerate(DoG(R0, sigma0, alpha)):
        mat[i,i] = a
    return spsolve(mat.tocsc(), intensities)
