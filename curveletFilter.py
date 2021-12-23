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
from listToCurv import *
import gc
import tensorflow as tf
import json

from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def scaleThat(inputDoidao):
    scales = []
    for i in range(len(inputDoidao)):
        scales.append(RobustScaler())
        inputDoidao[i] = scales[i].fit_transform(inputDoidao[i].reshape(-1,inputDoidao[i].shape[-1])).reshape(inputDoidao[i].shape)
    return scales

def unscaleThat(inputDoidao,scales):
    for i in range(len(inputDoidao)):
        inputDoidao[i] = scales[i].inverse_transform(inputDoidao[i].reshape(-1,inputDoidao[i].shape[-1])).reshape(inputDoidao[i].shape)

def ithasnan(arr):
    for element in arr:
        isnan = np.isnan(element)
        for _, val in np.ndenumerate(isnan):
            if val:
                return val
    return False

def threshold(arr,thresh):
    arr[arr < thresh] = 0
    return arr

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
experimentParam = param.string("param")

with open(experimentParam,'r') as arq:
    par = json.loads(arq.read())

patch_num       = par['patch_num']
patch_size      = par['patch_size']
stride_z        = par['stride_z']
stride_x        = par['stride_x']
val_split       = par['val_split']
lr              = par['lr']
nbscales        = par['nbscales']
nbangles_coarse = par['nbangles_coarse']

# patch_num = 400
# patch_size = 50
# stride_z = 20
# stride_x = 20
# val_split=0.2
# lr = 0.01
# nbscales=3
# nbangles_coarse=16

# X, Y = extract_patches(remigratedImg, migratedImg, patch_num, patch_size)
X, Y = remigratedImg, migratedImg
shape = X.shape
X = X.reshape(1,*shape,1)
Y = Y.reshape(1,*shape,1)

DCT = FDCT2D(shape, nbscales=nbscales, nbangles_coarse=nbangles_coarse)

c_X = DCT * X[0,:,:,0].ravel()
cr_X = DCT.struct(c_X)

contador = 0
shapes = []
for s in range(len(cr_X)):
    for w in range(len(cr_X[s])):
        shapes.append((*cr_X[s][w].shape,1))

inputDoidao  = [np.empty((1,*shape),dtype = np.float32) for shape in shapes]
outputDoidao = [np.empty((1,*shape),dtype = np.float32) for shape in shapes]

contador = 0
curvAux = DCT * X[0,:,:,0].ravel()
curvAux = DCT.struct(curvAux)
for s in range(len(curvAux)):
    for w in range(len(curvAux[s])):
        inputDoidao[contador][0,:,:,0] = threshold(np.abs(curvAux[s][w]),1e-6)
        contador += 1

contador = 0
curvAux = DCT * Y[0,:,:,0].ravel()
curvAux = DCT.struct(curvAux)
for s in range(len(curvAux)):
    for w in range(len(curvAux[s])):
        outputDoidao[contador][0,:,:,0] = threshold(np.abs(curvAux[s][w]),1e-6)
        contador += 1


scaleThat(inputDoidao)
scaleThat(outputDoidao)

model =  curvDomainFilter(shapes, learningRate = lr)
history = model.fit(inputDoidao, outputDoidao,
                    epochs=10,
                    callbacks=ClearMemory(),
                    batch_size=1)
                    # validation_split=0.2)

# serialize weights to HDF5
model.save_weights(f"weights/curvFilter_weights_lr_{lr}_{tag}.h5")
# print("Saved model weights to disk.")
sys.stderr.write("Saved model weights to disk.")
sys.stderr.flush()

# Save entire model (HDF5)
model.save(f"weights/curvFilter_lr_{lr}_{tag}.h5")
print("Saved model to disk.")
