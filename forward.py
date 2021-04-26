

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:22:17 2018

@author: david
"""

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib
import math
import matplotlib.pyplot as plt

#adjust the components of E0 so that they are orthogonal to d
    #if d is 2D, the z component is calculated such that ||[dx, dy, dz]||=1
def orthogonalize(E, d):
    if d.shape[-1] == 2:
        dz = math.sqrt(1 - d[..., 0]**2 - d[..., 1]**2)
        d = np.append(d, dz)
    s = np.cross(E, d)
    dE = np.cross(d, s)
    return dE
    #return dE/np.linalg.norm(dE) * np.linalg.norm(E)

def intensity(E):
    Econj = np.conj(E)
    I = np.sum(E*Econj, axis=-1)
    return np.real(I)



class layersample:
    
    #generate a layered sample based on a list of layer refractive indices
    #   layer_ri is a list of complex refractive indices for each layer
    #   z is the list of boundary positions along z
    def __init__(self, layer_ri, z):
        self.n = np.array(layer_ri).astype(np.complex128)
        self.z = np.array(z)
    
    #calculate the index of the field component associated with a layer
    #   l is the layer index [0, L)
    #   c is the component (x=0, y=1, z=2)
    #   d is the direction (0 = transmission, 1 = reflection)
    def i(self, l, c, d):
        i = l * 6 + d * 3 + c - 3
        return  i

    # generate the linear system corresponding to this layered sample and plane wave
    #   s is the direction vector scaled by the refractive index
    #   k0 is the free-space k-vector
    #   E is the field vector for the incident plane wave
    def generate_linsys(self, s, k0, E):
        #allocate space for the matrix
        L = len(self.n)
        M = np.zeros((6*(L-1), 6*(L-1)), dtype=np.complex128)
        
        #allocate space for the RHS vector
        b = np.zeros(6*(L-1), dtype=np.complex128)
        
        #initialize a counter for the equation number
        ei = 0
        
        #calculate the sz component for each layer
        self.sz = np.zeros(L, dtype=np.complex128)
        for l in range(L):
            self.sz[l] = np.sqrt(self.n[l]**2 - s[0]**2 - s[1]**2)
        
        #set constraints based on Gauss' law
        for l in range(0, L):
            #sz = np.sqrt(self.n[l]**2 - s[0]**2 - s[1]**2)
            
            #set the upward components for each layer
            #   note that layer L-1 does not have a downward component
            #   David I, Equation 7
            if l != L-1:
                M[ei, self.i(l, 0, 1)] = s[0]
                M[ei, self.i(l, 1, 1)] = s[1]
                M[ei, self.i(l, 2, 1)] = -self.sz[l]
                ei = ei+1
            
            #set the downward components for each layer
            #   note that layer 0 does not have a downward component
            #   Davis I, Equation 6
            if l != 0:
                M[ei, self.i(l, 0, 0)] = s[0]
                M[ei, self.i(l, 1, 0)] = s[1]
                M[ei, self.i(l, 2, 0)] = self.sz[l]
                ei = ei+1            
            
        #enforce a continuous field across boundaries
        for l in range(1, L):
            sz0 = self.sz[l-1]
            sz1 = self.sz[l]
            A = np.exp(1j * k0 * sz0 * (self.z[l] - self.z[l-1]))
            if l < L - 1:
                dl = self.z[l] - self.z[l+1]
                arg = -1j * k0 * sz1 * dl
                B = np.exp(arg)
            
            
            #if this is the second layer, use the simplified equations that account for the incident field
            if l == 1:
                M[ei, self.i(0, 0, 1)] = 1
                M[ei, self.i(1, 0, 0)] = -1
                if L > 2:
                    #print(-B, M[ei, self.i(1, 0, 1)])
                    M[ei, self.i(1, 0, 1)] = -B
                    
                b[ei] = -A * E[0]
                ei = ei + 1
                
                M[ei, self.i(0, 1, 1)] = 1
                M[ei, self.i(1, 1, 0)] = -1
                if L > 2:
                    M[ei, self.i(1, 1, 1)] = -B
                b[ei] = -A * E[1]
                ei = ei + 1
                
                M[ei, self.i(0, 2, 1)] = s[1]
                M[ei, self.i(0, 1, 1)] = sz0
                M[ei, self.i(1, 2, 0)] = -s[1]
                M[ei, self.i(1, 1, 0)] = sz1
                if L > 2:
                    M[ei, self.i(1, 2, 1)] = -B*s[1]
                    M[ei, self.i(1, 1, 1)] = -B*sz1
                b[ei] = A * sz0 * E[1] - A * s[1]*E[2]
                ei = ei + 1
                
                M[ei, self.i(0, 0, 1)] = -sz0
                M[ei, self.i(0, 2, 1)] = -s[0]
                M[ei, self.i(1, 0, 0)] = -sz1
                M[ei, self.i(1, 2, 0)] = s[0]
                if L > 2:
                    M[ei, self.i(1, 0, 1)] = B*sz1
                    M[ei, self.i(1, 2, 1)] = B*s[0]
                b[ei] = A * s[0] * E[2] - A * sz0 * E[0]
                ei = ei + 1
            
            #if this is the last layer, use the simplified equations that exclude reflections from the last layer
            elif l == L-1:
                M[ei, self.i(l-1, 0, 0)] = A
                M[ei, self.i(l-1, 0, 1)] = 1
                M[ei, self.i(l, 0, 0)] = -1
                ei = ei + 1
                
                M[ei, self.i(l-1, 1, 0)] = A
                M[ei, self.i(l-1, 1, 1)] = 1
                M[ei, self.i(l, 1, 0)] = -1
                ei = ei + 1
                
                M[ei, self.i(l-1, 2, 0)] = A*s[1]
                M[ei, self.i(l-1, 1, 0)] = -A*sz0
                M[ei, self.i(l-1, 2, 1)] = s[1]
                M[ei, self.i(l-1, 1, 1)] = sz0
                M[ei, self.i(l, 2, 0)] = -s[1]
                M[ei, self.i(l, 1, 0)] = sz1
                ei = ei + 1
                
                M[ei, self.i(l-1, 0, 0)] = A*sz0
                M[ei, self.i(l-1, 2, 0)] = -A*s[0]
                M[ei, self.i(l-1, 0, 1)] = -sz0
                M[ei, self.i(l-1, 2, 1)] = -s[0]
                M[ei, self.i(l, 0, 0)] = -sz1
                M[ei, self.i(l, 2, 0)] = s[0]
                ei = ei + 1
            #otherwise use the full set of boundary conditions
            else:
                M[ei, self.i(l-1, 0, 0)] = A
                M[ei, self.i(l-1, 0, 1)] = 1
                M[ei, self.i(l, 0, 0)] = -1
                M[ei, self.i(l, 0, 1)] = -B
                ei = ei + 1
                
                M[ei, self.i(l-1, 1, 0)] = A
                M[ei, self.i(l-1, 1, 1)] = 1
                M[ei, self.i(l, 1, 0)] = -1
                M[ei, self.i(l, 1, 1)] = -B
                ei = ei + 1
                
                M[ei, self.i(l-1, 2, 0)] = A*s[1]
                M[ei, self.i(l-1, 1, 0)] = -A*sz0
                M[ei, self.i(l-1, 2, 1)] = s[1]
                M[ei, self.i(l-1, 1, 1)] = sz0
                M[ei, self.i(l, 2, 0)] = -s[1]
                M[ei, self.i(l, 1, 0)] = sz1
                M[ei, self.i(l, 2, 1)] = -B*s[1]
                M[ei, self.i(l, 1, 1)] = -B*sz1
                ei = ei + 1
                
                M[ei, self.i(l-1, 0, 0)] = A*sz0
                M[ei, self.i(l-1, 2, 0)] = -A*s[0]
                M[ei, self.i(l-1, 0, 1)] = -sz0
                M[ei, self.i(l-1, 2, 1)] = -s[0]
                M[ei, self.i(l, 0, 0)] = -sz1
                M[ei, self.i(l, 2, 0)] = s[0]
                M[ei, self.i(l, 0, 1)] = B*sz1
                M[ei, self.i(l, 2, 1)] = B*s[0]
                ei = ei + 1
        
        return [M, b]
    
    #create a matrix for a single plane wave specified by k and E
    #   d = [dx, dy] are the x and y coordinates of the normalized direction of propagation
    #   k0 is the free space wave number (2 pi / lambda0)
    #   E is the electric field vector
    
    def solve1(self, d, k0, E):
                
        #s is the plane wave direction scaled by the refractive index
        s = np.array(d) * self.n[0]

               
        #store the matrix and RHS vector (for debugging)   
        [self.M, self.b] = self.generate_linsys(s, k0, E)
        #self.M = M
        #self.b = b
        
        #evaluate the linear system
        P = np.linalg.solve(self.M, self.b)
        
        #save the results (also for debugging)
        self.P = P
        
        #store the coefficients for each layer
        L = len(self.n)                             # calculate the number of layers
        self.Pt = np.zeros((3, L), np.complex128)
        self.Pr = np.zeros((3, L), np.complex128)
        for l in range(L):
            if l == 0:
                self.Pt[:, 0] = [E[0], E[1], E[2]]
            else:
                px = P[self.i(l, 0, 0)]
                py = P[self.i(l, 1, 0)]
                pz = P[self.i(l, 2, 0)]
                self.Pt[:, l] = [px, py, pz]
                
            if l == L-1:
                self.Pr[:, L-1] = [0, 0, 0]
            else:
                px = P[self.i(l, 0, 1)]
                py = P[self.i(l, 1, 1)]
                pz = P[self.i(l, 2, 1)]
                self.Pr[:, l] = [px, py, pz]
        
        #store values required for evaluation
        #store k
        self.k = k0
        
        #store sx and sy
        self.s = np.array([s[0], s[1]])
        
        self.solved = True
    
    #evaluate a solved homogeneous substrate
    def evaluate(self, X, Y, Z):
        
        if not self.solved:
            print("ERROR: the layered substrate hasn't been solved")
            return
        
        #this code is a bit cumbersome and could probably be optimized
        #   Basically, it vectorizes everything by creating an image
        #   such that the value at each pixel is the corresponding layer
        #   that the pixel resides in. That index is then used to calculate
        #   the field within the layer
        
        #allocate space for layer indices
        LI = np.zeros(Z.shape, dtype=np.int)
                
        #find the layer index for each sample point
        L = len(self.z)
        LI[Z < self.z[0]] = 0
        for l in range(L-1):
            idx = np.logical_and(Z > self.z[l], Z <= self.z[l+1])
            LI[idx] = l
            LI[Z > self.z[-1]] = L - 1
        
        #calculate the appropriate phase shift for the wave transmitted through the layer
        Ph_t = np.exp(1j * self.k * self.sz[LI] * (Z - self.z[LI]))
        
        #calculate the appropriate phase shift for the wave reflected off of the layer boundary
        LIp = LI + 1
        LIp[LIp >= L] = 0
        Ph_r = np.exp(-1j * self.k * self.sz[LI] * (Z - self.z[LIp]))
#        print(Ph_r)
        Ph_r[LI >= L-1] = 0
        
        #calculate the phase shift based on the X and Y positions
        Ph_xy = np.exp(1j * self.k * (self.s[0] * X + self.s[1] * Y))
        
        #apply the phase shifts
        Et = self.Pt[:, LI] * Ph_t[:, :]
        Er = self.Pr[:, LI] * Ph_r[:, :]
        
        #add everything together coherently
        E = (Et + Er) * Ph_xy[:, :]
        
        #return the electric field
        return np.moveaxis(E, 0, -1)


#This sample code produces a field similar to Figure 2 in Davis et al., 2010
def example_layer(n, M, filename):
    
    #set the material properties
    z_pos = [-100, 50]
    depths = np.linspace(z_pos[0], z_pos[1], len(n))  # specify the refractive indices of each layer
    #create a layered sample
    layers = layersample(n, depths)
    
    #set the input light parameters
    d = np.array([1, 0])                             #direction of propagation of the plane wave
    
    #d = d / np.linalg.norm(d)                           #normalize this direction vector
    l0 = 5                                              #specify the wavelength in free-space         
    k0 = 2 * np.pi / l0                              #calculate the wavenumber in free-space
    E0 = [0.7, 0, -0.7]                                      #specify the E vector
    E0 = orthogonalize(E0, d)                           #make sure that both vectors are orthogonal
    #solve for the substrate field
    
    layers.solve1(d, k0, E0)
    #set the simulation domain
    N = M
    D = [z_pos[0], z_pos[1]+50, 0, 0.5*(z_pos[1]-z_pos[0])]
    x = np.linspace(D[2], D[3], N)
    z = np.linspace(D[0], D[1], M)
    [X, Z] = np.meshgrid(x, z)
    Y = np.ones(X.shape) * 50
    E = layers.evaluate(X, Y, Z)
    Er = np.real(E)
    I = intensity(Er)
    print(np.average(I[-1,:]))
    # Write result to .txt file.
    with open(filename, "w") as f:
        for i in range(N):
            f.write(str(I[i, 0]))
            f.write('\n')
    # I = intensity(E)
    plt.figure()
    plt.set_cmap("afmhot")
    matplotlib.rcParams.update({'font.size': 32})
    # plt.subplot(1, 4, 1)
    plt.imshow(I, extent=(D[3], D[2], D[1], D[0]))
    # plt.title("Ex")
    # plt.subplot(1, 4, 2)
    # plt.imshow(Er[..., 1], extent=(D[3], D[2], D[1], D[0]))
    # plt.title("Ey")
    # plt.subplot(1, 4, 3)
    # plt.imshow(Er[..., 2], extent=(D[3], D[2], D[1], D[0]))
    # plt.title("Ez")
    # plt.subplot(1, 4, 4)
    # plt.imshow(I, extent=(D[3], D[2], D[1], D[0]))
    # plt.title("I")
    plt.show()

filename = 'd_obs.txt'

# example_layer(np.array([1,1.1,1.2,1.3,1.3,1.2]), 16, filename)
example_layer(np.array([1,2,2,1.5]), 64, filename)