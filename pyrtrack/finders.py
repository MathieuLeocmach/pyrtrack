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
import time
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from scipy.ndimage.morphology import grey_erosion, grey_dilation, binary_dilation
from scipy.ndimage import measurements
import numexpr

coefprime = np.array([1,-8, 0, 8, -1])
coefsec = np.array([-1, 16, -30, 16, -1])


def get_deconv_kernel(im, k=1.6, pxZ = 1.0, pxX=1.0):
    """Compute the deconvolution kernel from a priori isotropic image.
    Returned kernel is in Fourier space."""
    assert im.ndim == 3
    imbl = gaussian_filter(im, k)
    sblx = (np.abs(np.fft.rfft(imbl, axis=2))**2).mean(0).mean(0)
    sblz = (np.abs(np.fft.rfft(imbl, axis=0))**2).mean(1).mean(1)
    f2 = np.interp(
        np.fft.fftfreq(2*len(sblz), pxZ)[:len(sblz)],
        np.fft.fftfreq(2*len(sblx), pxX)[:len(sblx)], sblx
        )/sblz
    return np.sqrt(f2)

def deconvolve(im, kernel):
    """Deconvolve the input image. Suppose no noise (already blurred input)."""
    return np.fft.irfft(np.fft.rfft(im, axis=0) * kernel[:,None,None], axis=0, n=im.shape[0])

class CrockerGrierFinder:
    """A single scale blob finder using Crocker & Grier algorithm"""
    def __init__(self, shape=(256,256), dtype=np.float32):
        """Allocate memory once"""
        self.blurred = np.empty(shape, dtype)
        self.background = np.empty(shape, dtype)
        self.dilated = np.empty_like(self.blurred)
        self.binary = np.empty(self.blurred.shape, bool)

    def fill(self, image, k=1.6, uniform_size=None, background=None):
        """All the image processing when accepting a new image."""
        assert self.blurred.shape == image.shape, """Wrong image size:
%s instead of %s"""%(image.shape, self.blurred.shape)
        #fill the first layer by the input
        self.blurred[:] = image
        #Gaussian filter
        gaussian_filter(self.blurred, k, output=self.blurred)
        #background removal
        if background is None:
            if uniform_size is None:
                uniform_size = int(10*k)
            if uniform_size>0:
                uniform_filter(self.blurred, uniform_size, output=self.background)
                self.blurred -= self.background
        else:
            self.blurred -= background
        #Dilation
        grey_dilation(self.blurred, [3]*self.blurred.ndim, output=self.dilated)

    def initialize_binary(self, maxedge=-1, threshold=None):
        if threshold is None:
            self.binary[:] = self.blurred == self.dilated
        else:
            self.binary = numexpr.evaluate(
                '(b==d) & (b>thr)',
                {'b':self.blurred, 'd':self.dilated, 'thr':threshold}
                )
        #eliminate particles on the edges of image
        for a in range(self.binary.ndim):
            self.binary[tuple([slice(None)]*(self.binary.ndim-1-a)+[slice(0,2)])]=False
            self.binary[tuple([slice(None)]*(self.binary.ndim-1-a)+[slice(-2, None)])]=False
        #eliminate blobs that are edges
        if self.blurred.ndim==2 and maxedge>0 :
            for p in np.transpose(np.where(self.binary)):
                #xy neighbourhood
                ngb = self.blurred[tuple([slice(u-1, u+2) for u in p])]
                #compute the XYhessian matrix coefficients
                hess = [
                    ngb[0, 1] - 2*ngb[1, 1] + ngb[-1,1],
                    ngb[1, 0] - 2*ngb[1, 1] + ngb[1,-1],
                    ngb[0,0] + ngb[-1,-1] - ngb[0,-1] - ngb[-1,0]
                    ]
                #determinant of the Hessian, for the coefficient see
                #H Bay, a Ess, T Tuytelaars, and L Vangool,
                #Computer Vision and Image Understanding 110, 346-359 (2008)
                detH = hess[0]*hess[1] - hess[2]**2
                ratio = (hess[0]+hess[1])**2/(4.0*hess[0]*hess[1])
                if detH<0 or ratio>maxedge:
                    self.binary[tuple(p.tolist())] = False

    def no_subpix(self):
        """extracts centers positions and values from binary without subpixel resolution"""
        nb_centers = self.binary.sum()
        if nb_centers==0 or self.binary.min():
            return np.zeros([0, self.blurred.ndim+1])
        #original positions of the centers
        c0 = np.transpose(np.where(self.binary))
        vals = self.blurred[self.binary]
        return np.column_stack((vals, c0))

    def subpix(self):
        """Extract and refine to subpixel resolution the positions and size of the blobs"""
        nb_centers = self.binary.sum()
        if nb_centers==0 or self.binary.min():
            return np.zeros([0, self.blurred.ndim+1])
        centers = np.empty([nb_centers, self.blurred.ndim+1])
        #original positions of the centers
        c0 = np.transpose(np.where(self.binary))
        for i, p in enumerate(c0):
            #neighbourhood
            ngb = self.blurred[tuple([slice(u-2, u+3) for u in p])]
            for dim in range(ngb.ndim):
                a = ngb[tuple([1]*(ngb.ndim-1-dim)+[slice(None)]+[1]*dim)]
                centers[i,dim+1] = p[dim] - np.dot(coefprime,a)/np.dot(coefsec,a)
            centers[i,0] = ngb.mean()
        return centers

    def __call__(self, image, k=1.6, maxedge=-1, threshold=None, uniform_size=None, background=None):
        """Locate bright blobs in an image with subpixel resolution.
Returns an array of (x, y, intensity)"""
        self.fill(image, k, uniform_size, background)
        self.initialize_binary(maxedge, threshold)
        centers = self.subpix()[:,::-1]
        return centers

class OctaveBlobFinder:
    """Locator of bright blobs in an image of fixed shape. Works on a single octave."""
    def __init__(self, shape=(256,256), nbLayers=3, dtype=np.float32):
        """Allocate memory once"""
        self.layersG = np.empty([nbLayers+3]+list(shape), dtype)
        self.layers = np.empty([nbLayers+2]+list(shape), dtype)
        self.eroded = np.empty_like(self.layers)
        self.binary = np.empty(self.layers.shape, bool)
        self.time_fill = 0.0
        self.time_subpix = 0.0
        self.ncalls = 0
        self.noutputs = 0
        self.sizes = np.empty(nbLayers)

    def get_iterative_radii(self, k):
        nbLayers = len(self.layersG)-3
        #target blurring radii
        sigmas = k*2**(np.arange(nbLayers+3)/float(nbLayers))
        #corresponding blob sizes and iterative blurring radii
        return np.rint(sigmas*np.sqrt(2)).astype(int), np.sqrt(np.diff(sigmas**2))

    def fill(self, image, k=1.6):
        """All the image processing when accepting a new image."""
        t0 = time.process_time()
        #total 220 ms
        assert self.layersG[0].shape == image.shape, """Wrong image size:
%s instead of %s"""%(image.shape, self.layersG[0].shape)
        #fill the first layer by the input (already blurred by k)
        self.layersG[0] = image
        self.sizes, sigmas_iter = self.get_iterative_radii(k)
        #Gaussian filters
        for l, layer in enumerate(self.layersG[:-1]):
            gaussian_filter(layer, sigmas_iter[l], output=self.layersG[l+1])
        #Difference of gaussians
        for l in range(len(self.layers)):
            self.layers[l] = self.layersG[l+1] - self.layersG[l]
        #Erosion 86.2 ms
        grey_erosion(self.layers, [3]*self.layers.ndim, output=self.eroded)
        #scale space minima, whose neighbourhood are all negative 10 ms
        self.time_fill += time.process_time()-t0

    def initialize_binary(self, maxedge=-1, first_layer=False, maxDoG=None):
        """Convert the DoG layers into the binary image, True at center position.

        Centers are local minima of the DoG with negative value
            if maxDog is None, the DoG value should be further from 0 than machine precision.
            else, the DoG value must be lower than maxDog
        Centers at the edge of the image are excluded.
        On 2D images, if maxedge is positive, elongated blobs are excluded if the ratio of the eignevalues of the Hessian matrix is larger than maxedge.
        Optionally, the local spatial minima in the first DoG layer can be considered as centers.
        """
        if maxDoG is None:
            maxDoG = 0
        #local minima in the DoG on both space and scale are obtained from erosion
        self.binary = numexpr.evaluate(
            '(l==e) & (l<maxDoG) & (l**2+1.0>1.0)',
            {
                'l': self.layers,
                'e': self.eroded,
                'maxDoG': maxDoG
                }
            )
        #If the first DoG layer is taken into account,
        #its centers are translated in the layer above.
        #Necessary for subpixel
        if first_layer:
            #if a local maximum in the Gaussian layer 1 is
            #within 2px of a potential (but doomed) center in the DoG layer 0
            #add it to binary layer 1
            self.binary[0] = binary_dilation(
                self.binary[0],
                np.ones([5]*(self.layers.ndim-1))
                )
            #local maxima of the 0th Gaussian layer, idem Crocker&Grier
            self.binary[0] &= grey_dilation(
                self.layersG[0],
                [3]*(self.layers.ndim-1)
                ) == self.layersG[0]
            self.binary[1] |= self.binary[0]
        #centers in the first and last layers are discarded
        #do not remove these lines or you break subpixel
        self.binary[0] = False
        self.binary[-1] = False
        #eliminate particles on the edges of image
        for r, bi in zip(self.sizes[1:-1], self.binary[1:-1]):
            for a in range(bi.ndim):
                bi[tuple([slice(None)]*(bi.ndim-1-a)+[slice(0,r)])]=False
                bi[tuple([slice(None)]*(bi.ndim-1-a)+[slice(-r, None)])]=False
        #eliminate blobs that are edges
        if self.layers.ndim==3 and maxedge>0 :
            for r, bi, layer in zip(self.sizes[1:-1], self.binary[1:-1], self.layers[1:-1]):
                for p in np.transpose(np.where(bi)):
                    #xy neighbourhood
                    ngb = layer[tuple([slice(u-1, u+2) for u in p])]
                    #compute the XYhessian matrix coefficients
                    hess = [
                        ngb[0, 1] - 2*ngb[1, 1] + ngb[-1,1],
                        ngb[1, 0] - 2*ngb[1, 1] + ngb[1,-1],
                        ngb[0,0] + ngb[-1,-1] - ngb[0,-1] - ngb[-1,0]
                        ]
                    #determinant of the Hessian, for the coefficient see
                    #H Bay, a Ess, T Tuytelaars, and L Vangool,
                    #Computer Vision and Image Understanding 110, 346-359 (2008)
                    detH = hess[0]*hess[1] - hess[2]**2
                    ratio = (hess[0]+hess[1])**2/(4.0*hess[0]*hess[1])
                    if detH<0 or ratio>maxedge:
                        bi[tuple(p.tolist())] = False

    def no_subpix(self):
        """extracts centers positions and values from binary without subpixel resolution"""
        nb_centers = self.binary.sum()
        if nb_centers==0 or self.binary.min():
            return np.zeros([0, self.layers.ndim+1])
        #original positions of the centers
        c0 = np.transpose(np.where(self.binary))
        vals = self.layers[self.binary]
        return np.column_stack((vals, c0))

    def subpix(self, method=1):
        """Extract and refine to subpixel resolution the positions and size of the blobs"""
        nb_centers = self.binary.sum()
        if nb_centers==0 or self.binary.min():
            return np.zeros([0, self.layers.ndim+1])
        centers = np.empty([nb_centers, self.layers.ndim+1])
        #original positions of the centers
        c0 = np.transpose(np.where(self.binary))
        if method==0:
            for i, p in enumerate(c0):
                #neighbourhood
                ngb = self.layers[tuple([slice(u-1, u+2) for u in p])]
                grad = np.asarray([
                    (n[-1]-n[0])/2.0
                    for n in [
                        ngb[tuple(
                            slice(None) if a==u else 1
                            for u in range(ngb.ndim)
                            )]
                        for a in range(ngb.ndim)]
                    ])
                hess = np.empty([ngb.ndim]*2)
                for a in range(ngb.ndim):
                    n = ngb[tuple([1]*(ngb.ndim-1-a)+[slice(None)]+[1]*a)]
                    hess[a,a] = n[-1]+n[0]-2*n[1]
                for a in range(ngb.ndim-1):
                    for b in range(a+1,ngb.ndim):
                        n = ngb[tuple(
                            slice(None) if u==a or u==b else 1
                            for u in range(ngb.ndim)
                            )]
                        hess[a,b] = (n[0,0] + n[-1,-1] - n[0,-1] - n[-1,0])/4.0
                        hess[b,a] = hess[a,b]
                dx = - np.dot(np.linalg.inv(hess), grad)
                centers[i,1:] = p + dx
                centers[i,0] = ngb[tuple([1]*ngb.ndim)]+0.5*np.dot(dx,grad)
        else:
            for i, p in enumerate(c0):
                #neighbourhood, three pixels in the scale axis,
                #but according to scale in space
                r = self.sizes[p[0]]
                rv = [1]+[r]*(self.layers.ndim-1)
                ngb = np.copy(self.layers[tuple(
                    [slice(p[0]-1,p[0]+2)]+[
                        slice(u-r, u+r+1) for u in p[1:]
                        ]
                    )])
                #label only the negative pixels
                labels = measurements.label(ngb<0)[0]
                lab = labels[tuple(rv)]
                #value
                centers[i,0] = measurements.mean(ngb, labels, [lab])
                #pedestal removal
                ped = measurements.maximum(ngb, labels, [lab])
                if ped!=self.layers[tuple(p.tolist())]: #except if only one pixel or uniform value
                    ngb -= ped
                #center of mass
                centers[i,1:] = (np.asanyarray(measurements.center_of_mass(
                    ngb, labels, [lab]
                    ))-rv)+p
                #the subscale resolution is calculated using only 3 pixels
                n = ngb[tuple(
                    [slice(None)]+[r]*(self.layers.ndim-1)
                    )].ravel()
                denom = (n[2] - 2 * n[1] + n[0])
                if (abs(denom)+1.0)**2 > 1.0:
                    centers[i,1] = p[0] - (n[2] - n[0]) / 2.0 / denom
                else: centers[i,1] = p[0]
        return centers

    def __call__(self, image, k=1.6, maxedge=-1, first_layer=False, maxDoG=None):
        """Locate bright blobs in an image with subpixel resolution.
Returns an array of (x, y, r, -intensity in scale space)"""
        self.ncalls += 1
        self.fill(image, k)
        self.initialize_binary(maxedge, first_layer, maxDoG)
        t0 = time.process_time()
        centers = self.subpix()[:,::-1]
        self.time_subpix += time.process_time() - t0
        #convert scale to size
        n = (len(self.layers)-2)
        centers[:,-2] = scale2radius(centers[:,-2], k, n, self.layers.ndim-1)
        self.noutputs += len(centers)
        return centers


class MultiscaleBlobFinder:
    """Locator of bright blobs in an image of fixed shape. Works on more than one octave, starting at octave -1."""
    def __init__(self, shape=(256,256), nbLayers=3, nbOctaves=3, dtype=np.float32, Octave0=True):
        """Allocate memory for each octave"""
        shapes = np.vstack([np.ceil([s*2.0**(Octave0-o) for s in shape]) for o in range(nbOctaves)]).astype(int)
        self.preblurred = np.empty(shapes[0], dtype)
        self.octaves = [
            OctaveBlobFinder(s, nbLayers, dtype)
            for s in shapes if s.min()>8
            ] #shortens the list of octaves if no blob can be detected in that small window
        if not Octave0:
            self.octaves.insert(0, OctaveBlobFinder([0]*len(shape), nbLayers, dtype))
        self.Octave0 = Octave0
        self.time = 0.0
        self.ncalls = 0

    def __call__(self, image, k=1.6, Octave0=True, removeOverlap=True, maxedge=-1, deconvKernel=None, first_layer=False, maxDoG=None):
        """Locate blobs in each octave and regroup the results"""
        if not self.Octave0:
            Octave0 = False
        self.ncalls += 1
        t0 = time.process_time()
        if len(self.octaves)==0:
            return np.zeros([0, image.ndim+2])
        #upscale the image for octave -1
        #halfbl = gaussian_filter(np.array(im, , k/2.0)
        if Octave0:
            im2 = np.copy(image)
            for a in range(image.ndim):
                im2 = np.repeat(im2, 2, a)
            #preblur octave -1
            gaussian_filter(im2, k, output=self.preblurred)
            #locate blobs in octave -1
            centers = [self.octaves[0](self.preblurred, k, maxedge, maxDoG=maxDoG)]
        else:
            centers = []
        if len(self.octaves)>1:
            if deconvKernel is not None:
                assert len(deconvKernel) == int(image.shape[0]//2+1)
                assert image.ndim == 3
                #deconvolve the Z direction by a precalculated kernel
                #To avoid noise amplification, the blurred image is deconvolved, not the raw one
                deconv = deconvolve(gaussian_filter(image.astype(float), k), deconvKernel)
                #remove negative values
                centers += [self.octaves[1](np.maximum(0, deconv), maxedge=maxedge, first_layer=first_layer, maxDoG=maxDoG)]
            else:
                centers += [self.octaves[1](gaussian_filter(image, k), maxedge=maxedge, first_layer=first_layer, maxDoG=maxDoG)]
        #subsample the -3 layerG of the previous octave
        #which is two times more blurred that layer 0
        #and use it as the base of new octave
        for o, oc in enumerate(self.octaves[2:]):
            centers += [oc(
                self.octaves[o+1].layersG[-3][
                    tuple([slice(None, None, 2)]*image.ndim)],
                k, maxedge, maxDoG=maxDoG
                )]
        #merge the results and scale the coordinates and sizes
        centers = np.vstack([
            c * ([2**(o-Octave0)]*(1+image.ndim)+[1])
            for o, c in enumerate(centers)
            ])
        if len(centers)<2:
            return centers
        if not removeOverlap:
            return centers
        #remove overlaping objects (keep the most intense)
        #scales in dim*N^2, thus expensive if many centers and in high dimensions
        #Using a spatial index may be faster
        out = []
        for i in centers[np.argsort(centers[:,-1])]:
            for j in out:
                if np.sum((i[:-2]-j[:-2])**2) < (i[-2]+j[-2])**2:
                    break
            else:
                out.append(i)
        self.time += time.process_time() - t0
        return np.vstack(out)
