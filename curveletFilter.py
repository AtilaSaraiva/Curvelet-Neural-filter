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
import json

from sklearn.preprocessing import StandardScaler, RobustScaler

def scaleThat(inputDoidao):
    scales = []
    for i in range(len(inputDoidao)):
        scales.append(RobustScaler())
        inputDoidao[i] = scales[i].fit_transform(inputDoidao[i].reshape(-1,inputDoidao[i].shape[-1])).reshape(inputDoidao[i].shape)
    return scales

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

X, Y = extract_patches(remigratedImg, migratedImg, patch_num, patch_size)

DCT = FDCT2D((patch_size, patch_size), nbscales=nbscales, nbangles_coarse=nbangles_coarse)

c_X = DCT * X[0,:,:,0].ravel()
cr_X = DCT.struct(c_X)




contador = 0
shapes = []
for s in range(len(cr_X)):
    for w in range(len(cr_X[s])):
        shapes.append((*cr_X[s][w].shape,1))

inputDoidao  = [np.empty((patch_num,*shape),dtype = np.float32) for shape in shapes]
outputDoidao = [np.empty((patch_num,*shape),dtype = np.float32) for shape in shapes]

for patch in range(patch_num):
    contador = 0
    curvAux = DCT * X[patch,:,:,0].ravel()
    curvAux = DCT.struct(curvAux)
    for s in range(len(curvAux)):
        for w in range(len(curvAux[s])):
            inputDoidao[contador][patch,:,:,0] = threshold(np.abs(curvAux[s][w]),1e-6)
            contador += 1

for patch in range(patch_num):
    contador = 0
    curvAux = DCT * Y[patch,:,:,0].ravel()
    curvAux = DCT.struct(curvAux)
    for s in range(len(curvAux)):
        for w in range(len(curvAux[s])):
            outputDoidao[contador][patch,:,:,0] = threshold(np.abs(curvAux[s][w]),1e-6)
            contador += 1


scaleThat(inputDoidao)
scaleThat(outputDoidao)

model =  curvDomainFilter(shapes, learningRate = lr)
history = model.fit(inputDoidao, outputDoidao,
                    epochs=20,
                    validation_split=0.2)

# serialize weights to HDF5
model.save_weights(f"weights/curvFilter_weights_lr_{lr}_{tag}.h5")
print("Saved model weights to disk.")

# Save entire model (HDF5)
model.save(f"weights/curvFilter_lr_{lr}_{tag}.h5")
print("Saved model to disk.")

# ===============================================================
    # contador = 0
    # for s in range(len(X_trainCurv_reshape)):
        # for w in range(len(X_trainCurv_reshape[s])):
            # listaJanelas.append(X_trainCurv_reshape[s][w])
            # contador+=1




# train_num = int(patch_num*(1-val_split))
# val_num = int(patch_num*val_split)
# X_train, Y_train = extract_patches(norm_remigratedImg, norm_migratedImg, train_num, patch_size)
# X_val, Y_val     = extract_patches(norm_remigratedImg, norm_migratedImg, val_num, patch_size)


# nbscales=6
# nbangles_coarse=16
# DCT = FDCT2D((patch_size, patch_size), nbscales=nbscales, nbangles_coarse=nbangles_coarse)
# teste = np.ones((patch_size, patch_size))
# testeC = DCT * teste.ravel()
# curvDomainSize = testeC.shape[0]

# X_trainCurv = np.zeros((train_num,curvDomainSize,1))
# Y_trainCurv = np.zeros((train_num,curvDomainSize,1))
# X_valCurv   = np.zeros((val_num,curvDomainSize,1))
# Y_valCurv   = np.zeros((val_num,curvDomainSize,1))

# model = curvDomainFilter(input_size = (curvDomainSize,1), learningRate = lr)

# # for patch in range(train_num):
    # # X_trainCurv[patch,:,0] = np.abs(DCT * X_train[patch,:,:,0].ravel())
    # # Y_trainCurv[patch,:,0] = np.abs(DCT * Y_train[patch,:,:,0].ravel())
# # for patch in range(val_num):
    # # X_valCurv[patch,:,0]   = np.abs(DCT * X_val[patch,:,:,0].ravel())
    # # Y_valCurv[patch,:,0]   = np.abs(DCT * Y_val[patch,:,:,0].ravel())

# X_trainCurv  = []
# for patch in range(train_num):
    # X_trainCurv[patch,:,0] = DCT * X_train[patch,:,:,0].ravel()
    # Y_trainCurv[patch,:,0] = DCT * Y_train[patch,:,:,0].ravel()
    # X_trainCurv_reshape = DCT.struct(X_trainCurv[patch,:,0])

    # listaJanelas = []

    # contador = 0
    # for s in range(len(X_trainCurv_reshape)):
        # for w in range(len(X_trainCurv_reshape[s])):
            # listaJanelas.append(X_trainCurv_reshape[s][w])
            # contador+=1

    # print(contador)
    # listaPatches.append(listaJanelas)

# 0,1
# 1,8
# 2,16
# 3,32
# 4,64


# epochs = 10
# history = model.fit(X_trainCurv, Y_trainCurv,
                    # epochs=epochs,
                    # batch_size=10,
                    # validation_data=(X_valCurv, Y_valCurv))


# # serialize weights to HDF5
# model.save_weights(f"weights/curvFilter_weights_lr_{lr}_epoch_{epochs}_{tag}.h5")
# print("Saved model weights to disk.")

# # Save entire model (HDF5)
# model.save(f"weights/curvFilter_lr_{lr}_epoch_{epochs}_{tag}.h5")
# print("Saved model to disk.")
















# inputDoidao, nbangles, phase = curvToList(cr_X)
# testeC = listToCurv(inputDoidao,nbangles,phase)
# testeC = DCT.vect(testeC)

# for i in range(cr_X[s][w].shape[0]):
    # for j in range(cr_X[s][w].shape[1]):
        # print(cr_X[s][w][i,j],testeC[s][w][i,j])


# for s in range(len(cr_X)):
    # for w in range(len(cr_X[s])):
        # # print(cr_X[s][w].shape,testeC[s][w].shape)
        # # print(type(cr_X[s][w]),type(testeC[s][w]))
        # print(cr_X[s][w]-testeC[s][w])
        # # print(inputDoidao[contador])
        # contador+=1

# for input,shape in zip(inputDoidao,shapes):
    # print(input.shape, shape)



# inputDoidao, inputPhase, _, shapes = curvToListPatch_prepare(cr_X, patch_num)
# outputDoidao, outputPhase, _, _ = curvToListPatch_prepare(cr_X, patch_num)

# for patch in range(len(inputDoidao)):
    # curvAux = DCT * X[patch,:,:,0].ravel()
    # curvAux = DCT.struct(curvAux)
    # curvToListPatch(curvAux,inputDoidao,inputPhase,patch)

# for patch in range(len(outputDoidao)):
    # curvAux = DCT * Y[patch,:,:,0].ravel()
    # curvAux = DCT.struct(curvAux)
    # curvToListPatch(curvAux,inputDoidao,outputPhase,patch)

# print(ithasnan(inputDoidao))
# scales = []
# for i in range(len(inputDoidao)):
    # # print(np.prod(array.shape),array.reshape(-1,array.shape[-1]).shape)
    # scales.append(RobustScaler())
    # inputDoidao[i] = scales[i].fit_transform(inputDoidao[i].reshape(-1,inputDoidao[i].shape[-1])).reshape(inputDoidao[i].shape)
    # # print(scales[i].fit_transform(inputDoidao[i].reshape(-1,inputDoidao[i].shape[-1])).reshape(inputDoidao[i].shape).shape)

# print(ithasnan(inputDoidao))
