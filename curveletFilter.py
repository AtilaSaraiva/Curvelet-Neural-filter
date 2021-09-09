import numpy as np
import m8r
from dataclasses import dataclass
from matplotlib import pyplot as plt
from curvelops import FDCT2D
import pylops
import sys


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


DCT = FDCT2D((geometry.nz, geometry.nx), nbscales=6)

remigratedImg *= -1

m1Curv = DCT * migratedImg.ravel()
m2Curv = DCT * remigratedImg.ravel()

treshHold = 1e-5

filtr = m1Curv * m2Curv.conj() / (m2Curv.conj() * m2Curv + 0.000001)
resultCurv = np.abs(filtr) * m1Curv
resultCurv[np.abs(resultCurv) < treshHold] = treshHold
result = DCT.H * resultCurv
result = result.reshape((geometry.nz, geometry.nx))
result = np.real(result)



# result = DCT.H * m1Curv
# result = result.reshape((geometry.nz, geometry.nx))
# result = np.real(result)


FfilteredImage = m8r.Output()
FfilteredImage.put("d1",geometry.dz)
FfilteredImage.put("d2",geometry.dx)
FfilteredImage.put("n1",geometry.nz)
FfilteredImage.put("n2",geometry.nx)
FfilteredImage.write(result.T)

