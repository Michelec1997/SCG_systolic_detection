#Import libraries to create the model
from tensorflow.keras.models import Model
from keras.layers import Input, Concatenate, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import UpSampling1D
from keras.layers import Activation


POOL_SIZE = 2
N_FILTERS1 = 16
N_FILTERS2 = 32
N_FILTERS3 = 48
N_FILTERS4 = 64

def encoding_block(inputs, filters, pool=True):
    x = Conv1D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if pool:
        p = MaxPooling1D(POOL_SIZE)(x)
        return x, p
    return x

def decoding_block(x, skip):
    u = UpSampling1D(POOL_SIZE)(skip)
    c = Concatenate()([u, x])
    x = encoding_block(c, 64, pool=False)

    return x



def build_model(wind_size, channels):
    """ Build Unet network with 2d input layer"""

    x0 = Input(shape=(None, channels))

    # x0 = Input(self.input_shape, name="input")

    # Encoder
    x1, p1 = encoding_block(x0, N_FILTERS1, pool=True)
    x2, p2 = encoding_block(p1, N_FILTERS2, pool=True)
    x3, p3 = encoding_block(p2, N_FILTERS3, pool=True)
    x4, p4 = encoding_block(p3, N_FILTERS4, pool=True)

    # Bridge
    b = encoding_block(p4, 128, pool=False)

    # Decoder
    u1 = UpSampling1D(POOL_SIZE)(b)
    c1 = Concatenate()([u1, x4])
    x5 = encoding_block(c1, N_FILTERS4, pool=False)

    u2 = UpSampling1D(POOL_SIZE)(x5)
    c2 = Concatenate()([u2, x3])
    x6 = encoding_block(c2, N_FILTERS3, pool=False)
    u3 = UpSampling1D(POOL_SIZE)(x6)
    c3 = Concatenate()([u3, x2])
    x7 = encoding_block(c3, N_FILTERS2, pool=False)

    u4 = UpSampling1D(POOL_SIZE)(x7)
    c4 = Concatenate()([u4, x1])
    x8 = encoding_block(c4, N_FILTERS1, pool=False)
    
    # Output layer
    output =Conv1D(filters=1, kernel_size=1, activation='sigmoid')(x8)   
    # output = Conv1D(num_classes, 1, padding="same")(x8)

    modelCNN = Model(inputs=x0, outputs=output)


    return modelCNN