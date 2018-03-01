from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Add
from keras.models import Model
from keras.optimizers import SGD , Adam, Adadelta

encoding_dim = 64

def convAE(input_dims):
    # Autoencoder model definition
    input_img = Input(shape=input_dims)

    x = Conv2D(16, (3,3), activation='relu', padding ='same')(input_img)
    # x = MaxPooling2D((2, 2), padding ='same')(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    x = MaxPooling2D((2, 2), padding ='same')(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    x = MaxPooling2D((2, 2), padding ='same')(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    # x = MaxPooling2D((2, 2), padding ='same')(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    x = MaxPooling2D((2, 2), padding ='same')(x)
    encoded = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    # encoded = MaxPooling2D((2, 2), padding ='same')(x)

    x = Conv2D(8, (3,3), activation='relu', padding ='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3,3), activation='relu', padding ='same')(x)
    x = Conv2D(16, (3,3), activation='relu', padding ='same')(x)
    # x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3,3), activation='sigmoid', padding ='same')(x)

    autoencoder = Model(input_img, decoded)
    adam = Adam(lr=0.001,clipnorm=5.0)
    autoencoder.compile(optimizer=adam, loss='mse')

    print(autoencoder.summary())
    return autoencoder, None, None


def newConvAE(input_dims):
    # Autoencoder model definition
    input_img = Input(shape=input_dims)

    skip1 = Conv2D(encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(input_img)
    x = Conv2D(2*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(skip1)
    skip2 = Conv2D(4*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(x)
    x = Conv2D(8*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(skip2)
    encoded = Conv2D(16*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(x)

    x = Conv2DTranspose(8*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(encoded)
    x = Conv2DTranspose(4*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(x)
    # add2 = Add()([x, skip2])
    x = Conv2DTranspose(2*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(x)
    x = Conv2DTranspose(encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(x)
    # add1 = Add()([x, skip1])
    decoded = Conv2DTranspose(1, (3,3), activation='sigmoid', strides=(3,3), padding ='same')(x)

    autoencoder = Model(input_img, decoded)
    adam = Adam(lr=0.001,clipnorm=5.0)
    autoencoder.compile(optimizer=adam, loss='mse')

    print(autoencoder.summary())
    return autoencoder, None, None


def splitConvAE(input_dims):
    # Autoencoder model definition
    input_img = Input(shape=input_dims)

    conv1 = Conv2D(encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(input_img)
    encoded1 = Conv2D(4*encoding_dim, (15,15), activation='relu', strides=(9,9), padding ='same')(conv1)

    conv2 = Conv2D(2*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(conv1)
    encoded2 = Conv2D(4*encoding_dim, (15,15), activation='relu', strides=(9,9), padding ='same')(conv2)

    conv3 = Conv2D(2*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(conv2)
    encoded3 = Conv2D(4*encoding_dim, (15,15), activation='relu', strides=(9,9), padding ='same')(conv3)

    decoded3 = Conv2DTranspose(4*encoding_dim, (15,15), activation='relu', strides=(9,9), padding ='same')(encoded3)
    deconv3 = Conv2DTranspose(2*encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(decoded3)

    decoded2 = Conv2DTranspose(2*encoding_dim, (15,15), activation='relu', strides=(9,9), padding ='same')(encoded2)
    add2 = Add()([deconv3, decoded2])
    deconv2 = Conv2DTranspose(encoding_dim, (3,3), activation='relu', strides=(3,3), padding ='same')(add2)

    decoded1 = Conv2DTranspose(encoding_dim, (15,15), activation='relu', strides=(9,9), padding ='same')(encoded1)
    add1 = Add()([deconv2, decoded1])
    decoded = Conv2DTranspose(1, (3,3), activation='sigmoid', strides=(3,3), padding ='same')(add1)

    autoencoder = Model(input_img, decoded)
    adam = Adam(lr=0.001,clipnorm=5.0)
    autoencoder.compile(optimizer=adam, loss='mse')

    print(autoencoder.summary())
    return autoencoder, None, None
