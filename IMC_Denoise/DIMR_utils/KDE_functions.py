# -*- coding: utf-8 -*-

import numpy as np
from numpy import matlib as mb
import math

def KernelDensityEstimation(data, Bandwidth = 1, NumPoints = 100):
    yData = data # Nx1
    u = Bandwidth
    nPoints = int(NumPoints)
    
    if yData.ndim == 1:
        yData = np.expand_dims(yData, axis = -1)
    weight = np.ones(np.shape(yData))
    weight = np.divide(weight, np.shape(yData)[0])
    weight = np.transpose(weight)
    
    kernelcutoff = 4
    ty = yData
    
    fout, xout = compute_pdf(nPoints, weight, kernelcutoff, u, ty)
    xout = xout.transpose()
    
    return fout, xout

def compute_pdf(m, weight, cutoff, u, ty):
    foldwidth = np.minimum(cutoff, 3)
    xi = compute_default_xi(ty, foldwidth, m, u)
    
    if xi.ndim == 1:
        xi = np.expand_dims(xi, axis = -1)
    
    if ty.ndim == 1:
        ty = np.expand_dims(ty, axis = -1)
    
    xout = xi
    txi = xi
    
    f = dokernel(txi, ty, u, weight, cutoff, xi)
    f = np.divide(f, u)
    
    fout = f
    xout = xi
    
    return fout, xout

def compute_default_xi(ty, foldwidth, m, u):
    ximin = np.min(ty) - foldwidth * u
    ximax = np.max(ty) + foldwidth * u
    xi = np.linspace(ximin, ximax, m)
    return xi

def dokernel(txi, ty, u, weight, cutoff, xi):
    blocksize = 3e+4
    m = int(np.shape(txi)[0])
    n = int(np.shape(ty)[0])
    
    if n*m <= blocksize:
        ftemp = np.ones(shape = (n, m))
        z = np.divide(mb.repmat(np.transpose(txi), n, 1) - mb.repmat(ty, 1, m), u)
        f = gaussian_kernel(z)
        ftemp = ftemp * f
        f = np.matmul(weight, ftemp)
        
    else:
        ty = np.sort(ty, axis = 0)
        idx = np.argsort(ty[:,0], axis = 0)
        weight = weight[:,idx]
        
        f = np.zeros(shape = (1, m))
        stxi = np.sort(txi, axis = 0)
        idx = np.argsort(txi[:,0], axis = 0)
        
        jstart = 0
        jend = 0
        halfwidth = cutoff*u
        
        for kk in range(0, m):
            lo = stxi[kk] - halfwidth
            while ty[jstart,:] < lo and jstart < n-1:
                jstart += 1
                
            hi = stxi[kk] + halfwidth
            jend = np.maximum(jend, jstart)
            while ty[jend,:] <= hi and jend < n-1:
                jend += 1
            
            nearby = np.array(range(jstart, jend+1, 1))
            
            z = np.divide(stxi[kk] - ty[nearby], u)
            
            fk = np.matmul(weight[:, nearby], gaussian_kernel(z))
            f[:,kk] = fk
            
        f[:,idx] = f
        
    return f

def gaussian_kernel(z):
    f = np.exp(-0.5 * np.square(z))/np.sqrt(2*math.pi)
    return f


# if __name__ == '__main__':
#     data = 2* np.ones([1000,1])
#     data[0:10] = 0
#     data[80:99] = 0
#     data[800:999] = 1
#     fout, xout = KernelDensityEstimation(data,1,100)