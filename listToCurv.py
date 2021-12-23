import numpy as np
def threshold(arr,thresh):
    arr[arr < thresh] = 0
    return arr

def curvToList(curvArray):
    shapes = []
    for s in range(len(curvArray)):
        for w in range(len(curvArray[s])):
            shapes.append((*curvArray[s][w].shape,1))
    inputDoidao = [np.empty((1,*shape),dtype = np.float32) for shape in shapes]
    phase       = [np.empty(shape[0:2],dtype = np.float32) for shape in shapes]
    nbangles    = []
    contador    = 0
    for s in range(len(curvArray)):
        nbangles.append(len(curvArray[s]))
        for w in range(len(curvArray[s])):
            inputDoidao[contador][0,:,:,0] = threshold(np.abs(curvArray[s][w]),1e-6)
            phase[contador][:,:]           = np.angle(curvArray[s][w])
            contador += 1

    return inputDoidao, nbangles, phase

def listToCurv(lista, nbangles, phase):
    curvArray = []
    contador = 0
    for nba in nbangles:
        wedges = []
        for angle in range(nba):
            wedges.append(lista[contador][0,:,:,0] * np.exp(np.cdouble(1j) * phase[contador]))
            contador+=1

        curvArray.append(wedges)

    return curvArray

def curvToListPatch_prepare(curvArray, patch_num):
    shapes = []
    for s in range(len(curvArray)):
        for w in range(len(curvArray[s])):
            shapes.append((*curvArray[s][w].shape,1))
    inputDoidao = [np.empty((patch_num,*shape),dtype = np.float32) for shape in shapes]
    phase       = [np.empty((patch_num,*shape[0:2]),dtype = np.double) for shape in shapes]


    nbangles    = []
    for s in range(len(curvArray)):
        nbangles.append(len(curvArray[s]))
    return inputDoidao, phase, nbangles, shapes


def curvToListPatch(curvArray,inputDoidao,phase,patch):
    contador    = 0
    for s in range(len(curvArray)):
        for w in range(len(curvArray[s])):
            inputDoidao[contador][patch,:,:,0] = np.abs(curvArray[s][w])
            phase[contador][patch,:,:]         = np.angle(curvArray[s][w])
            contador += 1


def listToCurvPatch(nbangles, inputDoidao, phase, patch):
    curvArray = []
    contador = 0
    for nba in nbangles:
        wedges = []
        for angle in range(nba):
            wedges.append(inputDoidao[contador][patch,:,:,0] * np.exp(np.cdouble(1j) * phase[contador][patch,:,:]))
            contador+=1

        curvArray.append(wedges)

    return curvArray
