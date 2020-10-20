from keras.models import Sequential, Model
from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform


def conv2Dnet(input_shape, num_classes, norm=None):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding = 'same', activation ='relu', kernel_initializer = glorot_uniform(seed=0),
                     input_shape=input_shape, kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', kernel_initializer = glorot_uniform(seed=0), kernel_constraint=norm,
                    bias_constraint=norm))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation = 'softmax', kernel_initializer = glorot_uniform(seed=0),
                    kernel_constraint=norm, bias_constraint=norm))

    return model


def conv2Dsimple(input_shape, num_classes, norm=None):
    model = Sequential()
    model.add(Conv2D(16, (1, 1), padding = 'same', activation ='relu', kernel_initializer = glorot_uniform(seed=0),
                     input_shape=input_shape, kernel_constraint=norm, bias_constraint=norm))
    model.add(Conv2D(32, (3, 3), padding = 'same', activation ='relu', kernel_initializer = glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((4,4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', kernel_initializer = glorot_uniform(seed=0), kernel_constraint=norm,
                    bias_constraint=norm))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation = 'softmax', kernel_initializer = glorot_uniform(seed=0),
                    kernel_constraint=norm, bias_constraint=norm))

    return model


def conv2Dsimple2(input_shape, num_classes, norm=None):
    model = Sequential()
    model.add(Conv2D(16, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     input_shape=input_shape, kernel_constraint=norm, bias_constraint=norm))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     input_shape=input_shape, kernel_constraint=norm, bias_constraint=norm))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer=glorot_uniform(seed=0),
                     kernel_constraint=norm, bias_constraint=norm))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer=glorot_uniform(seed=0), kernel_constraint=norm,
                    bias_constraint=norm))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0),
                    kernel_constraint=norm, bias_constraint=norm))

    return model




