import numpy as np
import keras
from keras.layers import Conv2D, Dropout
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
from keras.models import *


def curvDomainFilter(shapes,
                     pretrained_weights=None,
                     input_size=(32, 32),
                     learningRate=0.5e-4):
    inputs = [keras.Input(shape) for shape in shapes]
    outputs = []

    for input in inputs:
        conv1 = Conv2D(32,
                       3,
                       activation='tanh',
                       padding='same',
                       kernel_initializer='he_normal')(input)
        conv3 = Conv2D(1,
                       3,
                       activation='linear',
                       padding='same',
                       kernel_initializer='he_normal')(conv1)
        outputs.append(conv3)

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
