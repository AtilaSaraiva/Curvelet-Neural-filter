import numpy as np
import random
import m8r
from dataclasses import dataclass
from matplotlib import pyplot as plt
from curvelops import FDCT2D
import pylops
import sys
from keras.models import load_model
from tile_dataset import tile, merge
from curvModel import curvDomainFilter
from sklearn.preprocessing import StandardScaler, RobustScaler
from listToCurv import *
from tqdm import tqdm

def ithasnan(arr):
    for element in arr:
        isnan = np.isnan(element)
        for _, val in np.ndenumerate(isnan):
            if val:
                return val
    return False

@dataclass(frozen=True)
class Geometry:
    dz: float
    dx: float
    nz: int
    nx: int

def read_rsf (tag):
    imgFile = m8r.Input(tag = tag)
    nz      = imgFile.int("n1")
    nx      = imgFile.int("n2")

    geometry = Geometry(dz = imgFile.float("d1"),
                       dx = imgFile.float("d2"),
                       nz = nz,
                       nx = nx)

    fileArray = np.zeros(nz*nx,'f')
    imgFile.read(fileArray)

    inputShape = (nz, nx)
    fileArray = fileArray[0:(nz*nx)].reshape(inputShape, order="F")

    return fileArray, geometry


param = m8r.Par()

m1 = param.string("m1")
weights = param.string("weights")
migratedImg, geometry = read_rsf(m1)
model = load_model(weights)

migScaleModel = RobustScaler()
migScaleModel.fit(migratedImg)
norm_migratedImg = migScaleModel.transform(migratedImg)

patch_size = 100
stride_z = 80
stride_x = 80
val_split=0.2
lr = 0.0001

nbscales=6
nbangles_coarse=16
DCT = FDCT2D((patch_size, patch_size), nbscales=nbscales, nbangles_coarse=nbangles_coarse)

tiles_migratedImg = tile(norm_migratedImg, patch_size, stride_z, stride_x)
patch_num = tiles_migratedImg.shape[0]

c_X = DCT * tiles_migratedImg[0,:,:,0].ravel()
cr_X = DCT.struct(c_X)


resultTiles = np.zeros(tiles_migratedImg.shape)
for patch in tqdm(range(patch_num)):
    rtmCurv                      = DCT * tiles_migratedImg[patch,:,:,0].ravel()
    rtmCurv                      = DCT.struct(rtmCurv)
    inputDoidao, nbangles, phase = curvToList(rtmCurv)
    predictedAmp                 = model.predict(inputDoidao)
    print(patch,ithasnan(inputDoidao),ithasnan(predictedAmp))
    resultAux                    = listToCurv(predictedAmp,nbangles,phase)
    resultAux                    = DCT.vect(resultAux)
    resultAux                    = np.real(DCT.H * resultAux)
    resultTiles[patch,:,:,0]     = resultAux.reshape((patch_size,patch_size))

# resultTilesCurv = model.predict_on_batch(inputDoidao)


# for patch in range(patch_num):
    # resultAux                = listToCurvPatch(nbangles, resultTilesCurv, inputPhase, patch)
    # resultAux                = DCT.vect(resultAux)
    # resultAux                = DCT.H * resultAux
    # resultTiles[patch,:,:,0] = np.real(resultAux.reshape((patch_size,patch_size)))

# norm_result = merge(resultTiles, geometry.nz, geometry.nx, patch_size, stride_z, stride_x)
# result = migScaleModel.inverse_transform(norm_result[:,:,0])

# FfilteredImage = m8r.Output()
# FfilteredImage.put("d1",geometry.dz)
# FfilteredImage.put("d2",geometry.dx)
# FfilteredImage.put("n1",geometry.nz)
# FfilteredImage.put("n2",geometry.nx)
# FfilteredImage.write(result.T)


# DCT = FDCT2D((patch_size, patch_size), nbscales=6)
# teste = np.ones((patch_size, patch_size))
# testeC = DCT * teste.ravel()
# curvDomainSize = testeC.shape[0]


# tiles_migratedImg = tile(norm_migratedImg, patch_size, stride_z, stride_x)
# patch_num = tiles_migratedImg.shape[0]
# batches = np.zeros((patch_num,curvDomainSize,1))
# phases = np.zeros((patch_num,curvDomainSize))
# resultTiles = np.zeros(tiles_migratedImg.shape)

# resultTilesCurv = np.zeros((patch_num,curvDomainSize,1))
# sys.stderr.write('\n'+str(patch_num)+'\n')
# sys.stderr.flush()
# for patch in range(patch_num):
    # rtmCurv            = DCT * tiles_migratedImg[patch,:,:,0].ravel()
    # phases[patch,:]    = np.angle(rtmCurv)
    # # batches[patch,:,0] = np.abs(rtmCurv)
    # amplitude = np.abs(rtmCurv).reshape((1,curvDomainSize,1))

    # sys.stderr.write(f"iteration {patch}\n")
    # sys.stderr.flush()
    # resultTilesCurv[patch,:,:] = model.predict(amplitude)



# # resultTilesCurv = model.predict_on_batch(batches)

# for patch in range(patch_num):
    # aux = resultTilesCurv[patch,:,0] * np.exp(phases[patch,:] * 1j)
    # aux = DCT.H * aux
    # aux = aux.reshape((patch_size,patch_size))
    # resultTiles[patch,:,:,0] = aux

# norm_result = merge(resultTiles, geometry.nz, geometry.nx, patch_size, stride_z, stride_x)

# result = migScaleModel.inverse_transform(norm_result[:,:,0])

# FfilteredImage = m8r.Output()
# FfilteredImage.put("d1",geometry.dz)
# FfilteredImage.put("d2",geometry.dx)
# FfilteredImage.put("n1",geometry.nz)
# FfilteredImage.put("n2",geometry.nx)
# FfilteredImage.write(result.T)
