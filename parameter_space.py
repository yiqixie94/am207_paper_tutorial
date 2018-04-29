'''
some packaged methods for high-dimensional space analysis
author: Yiqi Xie, Rui Fang
'''


import os, sys
import numpy as np


class Line:
    '''a line in high dimensional space
        mainly help with the coordinate transformation'''

    def __init__(self, A, ex):
        '''create from a point and a direction'''
        A_ = np.array(A).reshape(-1)
        ex_ = np.array(ex).reshape(-1)
        self.A = A_
        self.ex = ex_
        self.dim = len(A_)
    def __repr__(self):
        return self.__class__.__name__\
                    + '(A={}, ex={})'\
                        .format(repr(self.A), repr(self.ex))
    @classmethod
    def from_AB(cls, A, B):
        '''create from two points'''
        ab = B - A
        ex = ab / np.linalg.norm(ab)
        return cls(A, ex)
    def recv(self, v):
        '''transform from 1d coordinate to full-space vector'''
        return self.A + np.dot(v, self.ex)
    def proj(self, V):
        '''transform from full-space vector to 1d coordinate'''
        return np.dot(self.ex, V - self.A)

    def spread(self, distgrid):
        '''spread the A point to generate a grid on the line'''
        new_params = []
        for t in distgrid:
            param = self.A + t * self.ex
            new_params.append(param)
        return new_params


class Plane:
    '''a plane in high dimensional space
        mainly help with the coordinate transformation'''
    
    def __init__(self, A, ex, ey):
        '''create from a point and 2 directions'''
        A_ = np.array(A).reshape(-1)
        ex_ = np.array(ex).reshape(-1)
        ey_ = np.array(ey).reshape(-1)
        self.A = A_
        self.exy = np.vstack([ex_, ey_])
        self.dim = len(A_)
    def __repr__(self):
        return self.__class__.__name__\
                    + '(A={}, ex={}, ey={})'\
                        .format(repr(self.A), repr(self.exy[0]), repr(self.exy[-1]))
    @classmethod
    def from_ABC(cls, A, B, C):
        '''create from three points'''
        ab = B - A
        ac = C - A
        ex = ab / np.linalg.norm(ab)
        ac_projy = ac - np.dot(ac, ex) * ex
        ey = ac_projy / np.linalg.norm(ac_projy)
        return cls(A, ex, ey)
    def recv(self, v):
        '''transform from 2d coordinate to full-space vector'''
        return self.A + np.dot(v, self.exy)
    def proj(self, V):
        '''transform from full-space vector to 2d coordinate'''
        return np.dot(self.exy, V - self.A)

    
    
class DimReducedMesh:
    '''a functionality supplement to the Plane class'''
    
    def __init__(self, plane, xs, ys):
        '''plane is a Plane object, xs and ys are inputs to numpy.meshgrid(...)'''
        x_mesh, y_mesh = np.meshgrid(xs, ys)
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.plane = plane
        print('initialized with mesh shape {1}-xs {0}-ys'.format(*x_mesh.shape))
    def mesh(self):
        return self.x_mesh, self.y_mesh
    def meshrecv(self):
        '''transform the 2d mesh to a list of full-space vectors'''
        coords = np.vstack([self.x_mesh.reshape(-1), self.y_mesh.reshape(-1)]).T # dim=(ny*nx, 2)
        Vs = []
        for i, xy in enumerate(coords):
            print('\rrecovering {}/{}...'.format(i+1, len(coords)))
            Vs.append(self.plane.recv(xy)) # dim(ny*nx, N)
        print('done')
        return Vs
    def resreshape(self, results):
        '''reshape the mesh evaluations to the shape of the mesh'''
        return np.array(results).reshape(self.x_mesh.shape)
    

    

class RandomDirections:
    '''random directions in the full-space'''

    def __init__(self, directions):
        self.directions = directions # shape (n_dir, dim)
        self.dim = directions.shape[1]
        self.n = directions.shape[0]

    @classmethod
    def from_npy(cls, path):
        '''load from numpy formatted file'''
        directions = np.load(path)
        return cls(directions)

    @classmethod
    def generate(cls, dim, n, batch=None, seed=None):
        '''generate `n` random directions in a `dim`-dimensional space
            use `batch` as batch size to control the porcess 
            (as it could be highly compute-intensive)
        '''
        if batch is None:
            batch = self.dim
        if seed is not None:
            np.random.seed(seed)

        sizes = []
        while dim > batch:
            dim -= batch
            sizes.append(batch)
        sizes.append(dim)

        mvns = []
        for i, s in enumerate(sizes):
            sys.stdout.flush()
            print('\rprocessing dim batch {}/{}...'.format(i+1, len(sizes)), end='')
            mean, cov = np.zeros(s), np.identity(s)
            mvns.append(np.random.multivariate_normal(mean, cov, n))
        print('done')

        mvns = np.hstack(mvns)
        for i in range(n):
            print('\rnormalizing direction {}/{}...'.format(i+1, n), end='')
            norm = np.linalg.norm(mvns[i])
            mvns[i] /= norm
        print('done')

        return cls(mvns)

    def spread(self, center, idir, distgrid):
        '''use the `idir`-th direction to spread the ray from `center` 
            with given `distgrid` as the distance grid'''
        new_params = []
        for t in distgrid:
            param = center + t * self.directions[idir]
            new_params.append(param)
        return new_params