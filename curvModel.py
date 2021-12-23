import numpy as np
import keras
from keras import regularizers
import sys
# from keras.layers import Conv2D, Dropout

from keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# from keras.optimizers import Adam
from keras.models import *


def curvDomainFilter(shapes,
                     pretrained_weights=None,
                     input_size=(32, 32),
                     learningRate=0.5e-4):
    inputs = [keras.Input(shape) for shape in shapes]
    outputs = []

    for input in inputs:
        maxLen = min(input.shape[1],input.shape[2])
        unetBlocks = min(int(np.log2(maxLen) - 1),3)
        if unetBlocks >= 1:
            pool = input
            featMaps = 16
            convLayers = []
            for n in range(unetBlocks):
                convLayers.append(Conv2D(featMaps,
                                  3,
                                  activation='tanh',
                                  padding='same',
                                  kernel_initializer='he_normal')(pool))
                                  # activity_regularizer=regularizers.l1(1e-4))(pool))
                pool = MaxPooling2D(pool_size=(2, 2))(convLayers[n])
                featMaps *= 2
            merge = pool
            featMaps = int(featMaps/2)
            for n in range(unetBlocks):
                conv = Conv2D(featMaps,
                              3,
                              activation='tanh',
                              padding='same',
                              kernel_initializer='he_normal')(merge)
                              # activity_regularizer=regularizers.l1(1e-4))(merge)
                up = tf.image.resize(conv,convLayers[-1].shape[1:3].as_list())
                merge = concatenate([up,convLayers.pop(-1)], axis = 3)
                featMaps = int(featMaps/2)

            conv = Conv2D(1,
                          3,
                          activation='linear',
                          padding='same',
                          kernel_initializer='he_normal',
                          activity_regularizer=regularizers.l1(1e-4))(merge)
            outputs.append(conv)
        else:
            outputs.append(input)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=learningRate),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

if __name__ == "__main__":
    inputs = [np.zeros((4, 32, 32, 1)) for i in range(5)]
    shapes = [input.shape[-3:] for input in inputs]
    outputs = [np.zeros((4, 32, 32, 1)) for i in range(5)]

    lr=0.0001
    model =  curvDomainFilter(shapes, learningRate = lr)
    history = model.fit(inputs, outputs,
                        epochs=2,
                        validation_split=0.5)


# print(curvDomainFilter(shapes))
    # numPatches = 4
    # listaPatchesInput  = []
    # listaPatchesOutput = []
    # for patch in range(numPatches):
        # listaAuxiliarInput  = []
        # listaAuxiliarOutput = []
        # for it in range(10):
            # listaAuxiliarInput.append(np.zeros((32, 32, 1)))
            # listaAuxiliarOutput.append(np.zeros((32, 32, 1)))

        # listaPatchesInput.append(listaAuxiliarInput)
        # listaPatchesOutput.append(listaAuxiliarOutput)

        # if(patch == 0):
            # shapes = [input.shape for input in listaPatchesInput[0]]
            # # shapes = [print(input) for input in listaPatchesInput[0]]
            # # for i in range(len(listaPatchesInput[0])):
                # # print(i)

    # print(shapes)

    # # inputs = [np.zeros((4, 32, 32, 1)) for i in range(5)]
    # # outputs = [np.zeros((4, 32, 32, 1)) for i in range(5)]

    # lr=0.0001
    # model =  curvDomainFilter(shapes, learningRate = lr)
    # # history = model.fit(inputs, outputs,
                        # # epochs=2,
                        # # validation_split=0.5)
    # history = model.fit(listaPatchesInput, listaPatchesOutput,
                        # epochs=2,
                        # validation_split=0.5)

# # print(curvDomainFilter(shapes))
