import numpy as np
import m8r
from dataclasses import dataclass
from matplotlib import pyplot as plt
from curvelops import FDCT2D
import pylops
import sys
from tile_dataset import tile, merge

from sklearn.preprocessing import StandardScaler, RobustScaler


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
migratedImg, geometry = read_rsf(m1)
m2 = param.string("m2")
remigratedImg, geometry = read_rsf(m2)

migScaleModel = RobustScaler()
migScaleModel.fit(migratedImg)
norm_migratedImg = migScaleModel.transform(migratedImg)

remigScaleModel = RobustScaler()
remigScaleModel.fit(remigratedImg)
norm_remigratedImg = remigScaleModel.transform(remigratedImg)

patch_num = 1000
patch_size = 100
stride_z = 20
stride_x = 20


tiles_migratedImg = tile(norm_migratedImg, patch_size, stride_z, stride_x)
tiles_remigratedImg = tile(norm_remigratedImg, patch_size, stride_z, stride_x)


patch_num = tiles_migratedImg.shape[0]
if tiles_remigratedImg.shape[0] != patch_num:
    print("shapes dos inputs incompativeis")
    exit


tiles_results = np.zeros(tiles_migratedImg.shape)

DCT = FDCT2D((patch_size, patch_size), nbscales=6)
for patch in range(patch_num):
    tile_migratedImg = tiles_migratedImg[patch,:,:,0]
    tile_remigratedImg = tiles_remigratedImg[patch,:,:,0]


    # remigratedImg *= -1

    m1Curv = DCT * tile_migratedImg.ravel()
    m2Curv = DCT * tile_remigratedImg.ravel()

    # treshHold = 1e-5

    filtr = m1Curv * m2Curv.conj() / (m2Curv.conj() * m2Curv)
    resultCurv = np.abs(filtr) * m1Curv
    # resultCurv[np.abs(resultCurv) < treshHold] = treshHold
    result = DCT.H * resultCurv
    result = result.reshape((patch_size, patch_size))
    tiles_results[patch,:,:,0] = np.real(result)

    # result = DCT.H * m1Curv
    # result = result.reshape((geometry.nz, geometry.nx))
    # result = np.real(result)

result = merge(tiles_results, geometry.nz, geometry.nx, patch_size, stride_z, stride_x)

FfilteredImage = m8r.Output()
FfilteredImage.put("d1",geometry.dz)
FfilteredImage.put("d2",geometry.dx)
FfilteredImage.put("n1",geometry.nz)
FfilteredImage.put("n2",geometry.nx)
FfilteredImage.write(result.T)
