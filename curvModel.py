from keras.layers import Conv1D, Dropout
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
from keras.models import *

def curvDomainFilter(pretrained_weights = None, input_size = (512,1), learningRate=0.5e-4):
    inputs = Input(input_size)
    # conv3 = Conv1D(32, 1, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv1D(32, 1, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv3 = Conv1D(1, 1, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # conv2 = Conv1D(32, 1, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # conv3 = Conv1D(1, 1, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    model = Model(inputs = inputs, outputs = conv3)

    model.compile(optimizer = Adam(learning_rate = learningRate),
            loss = 'mean_squared_error',
            metrics = ['mean_squared_error'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
