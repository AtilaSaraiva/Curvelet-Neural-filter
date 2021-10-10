import numpy as np
import random
from curvelops import FDCT2D
import pylops
from tile_dataset import tile, merge
from curvModel import curvDomainFilter
from listToCurv import curvToList, listToCurv

print('Iniciando teste das funções curvToList e listToCurv')

nz=1000
nx=1000
nbscales=6
nbangles_coarse=16
DCT = FDCT2D((nz, nx), nbscales=nbscales, nbangles_coarse=nbangles_coarse)
X = np.random.random((nz,nx))
c_X = DCT * X.ravel()
cr_X = DCT.struct(c_X)
inputDoidao, nbangles, phase = curvToList(cr_X)
testeC = listToCurv(inputDoidao,nbangles,phase)
testeC = DCT.vect(testeC)
teste = DCT.H * testeC
teste = teste.reshape((nz,nx))


error = 0.
for i in range(nz):
    for j in range(nx):
        error += np.abs(X[i,j] - teste[i,j])

error = error / (nz*nx)

print(error)

if error < 1e-8:
    print('OK')

