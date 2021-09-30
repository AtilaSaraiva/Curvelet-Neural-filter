import numpy as np
import random
import m8r
from dataclasses import dataclass
from matplotlib import pyplot as plt
from curvelops import FDCT2D
import pylops
import sys
from tile_dataset import tile, merge
from curvModel import curvDomainFilter

from sklearn.preprocessing import StandardScaler, RobustScaler


def extract_patches(data, mask, patch_num, patch_size):

    X = np.empty((patch_num, patch_size, patch_size,1))
    Y = np.empty((patch_num, patch_size, patch_size,1))

    (z_max, x_max) = data.shape

    for n in range(patch_num):

        # Select random point in data (not too close to edge)
        x_n = random.randint(patch_size // 2, x_max - patch_size // 2)
        z_n = random.randint(patch_size // 2, z_max - patch_size // 2)

        # Extract data and mask patch around point
        X[n,:,:,0] = data[z_n-patch_size//2:z_n+patch_size//2,x_n-patch_size//2:x_n+patch_size//2]
        Y[n,:,:,0] = mask[z_n-patch_size//2:z_n+patch_size//2,x_n-patch_size//2:x_n+patch_size//2]


    return X, Y

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
tag = param.string("tag")
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
val_split=0.2
lr = 0.0001


train_num = int(patch_num*(1-val_split))
val_num = int(patch_num*val_split)
# tiles_migratedImg = tile(norm_migratedImg, patch_size, stride_z, stride_x)
# tiles_remigratedImg = tile(norm_remigratedImg, patch_size, stride_z, stride_x)
X_train, Y_train = extract_patches(norm_remigratedImg, norm_migratedImg, train_num, patch_size)
X_val, Y_val     = extract_patches(norm_remigratedImg, norm_migratedImg, val_num, patch_size)


DCT = FDCT2D((patch_size, patch_size), nbscales=6)
teste = np.ones((patch_size, patch_size))
testeC = DCT * teste.ravel()
curvDomainSize = testeC.shape[0]

X_trainCurv = np.zeros((train_num,curvDomainSize,1))
Y_trainCurv = np.zeros((train_num,curvDomainSize,1))
X_valCurv   = np.zeros((val_num,curvDomainSize,1))
Y_valCurv   = np.zeros((val_num,curvDomainSize,1))

model = curvDomainFilter(input_size = (curvDomainSize,1), learningRate = lr)

for patch in range(train_num):
    X_trainCurv[patch,:,0] = np.abs(DCT * X_train[patch,:,:,0].ravel())
    Y_trainCurv[patch,:,0] = np.abs(DCT * Y_train[patch,:,:,0].ravel())
for patch in range(val_num):
    X_valCurv[patch,:,0]   = np.abs(DCT * X_val[patch,:,:,0].ravel())
    Y_valCurv[patch,:,0]   = np.abs(DCT * Y_val[patch,:,:,0].ravel())


epochs = 10
history = model.fit(X_trainCurv, Y_trainCurv,
					epochs=epochs,
					batch_size=10,
					validation_data=(X_valCurv, Y_valCurv))


# serialize weights to HDF5
model.save_weights(f"weights/curvFilter_weights_lr_{lr}_epoch_{epochs}_{tag}.h5")
print("Saved model weights to disk.")

# Save entire model (HDF5)
model.save(f"weights/curvFilter_lr_{lr}_epoch_{epochs}_{tag}.h5")
print("Saved model to disk.")


    # # treshHold = 1e-5

    # filtr = m1Curv * m2Curv.conj() / (m2Curv.conj() * m2Curv)
    # resultCurv = np.abs(filtr) * m1Curv
    # # resultCurv[np.abs(resultCurv) < treshHold] = treshHold
    # result = DCT.H * resultCurv
    # result = result.reshape((patch_size, patch_size))
    # tiles_results[patch,:,:,0] = np.real(result)

    # # result = DCT.H * m1Curv
    # # result = result.reshape((geometry.nz, geometry.nx))
    # # result = np.real(result)

# result = merge(tiles_results, geometry.nz, geometry.nx, patch_size, stride_z, stride_x)

# FfilteredImage = m8r.Output()
# FfilteredImage.put("d1",geometry.dz)
# FfilteredImage.put("d2",geometry.dx)
# FfilteredImage.put("n1",geometry.nz)
# FfilteredImage.put("n2",geometry.nx)
# FfilteredImage.write(result.T)
