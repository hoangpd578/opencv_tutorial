import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, models
from tensorflow.keras import callbacks, regularizers


class Vgg16:
    @staticmethod
    def build(width, hight, ndim, classes):
        weight_decay = 0.000

        model = models.Sequential()
        model.add(layers.Reshape(target_shape= [96, 96, 3], input_shape= [width, hight, ndim]))

        #block1
        model.add(layers .Conv2D(64, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers. Conv2D(64, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers. BatchNormalization())
        model.add(layers .MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
        model.add(layers .Dropout(0.3))

        #block 2
        model.add(layers. Conv2D(128, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers. BatchNormalization())
        model.add(layers. Conv2D(128, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .BatchNormalization())
        model.add(layers. Dropout(0.25))
        model.add(layers .MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
        model.add(layers. BatchNormalization())
        model.add(layers .Dropout(0.25))

        #block 3
        model.add(layers .Conv2D(256, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .BatchNormalization())
        model.add(layers .Conv2D(256, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .BatchNormalization())
        model.add(layers. Conv2D(256, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .Dropout(0.3))
        model.add(layers .MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
        model.add(layers. BatchNormalization())
        model.add(layers .Dropout(0.3))

        #block4
        model.add(layers .Conv2D(512, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .BatchNormalization())
        model.add(layers. Conv2D(512, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .BatchNormalization())
        model.add(layers. Dropout(0.3))
        model.add(layers. Conv2D(512, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers. BatchNormalization())
        model.add(layers .MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
        model.add(layers. Dropout(0.4))

        #fully connected
        model.add(layers .Flatten())
        model.add(layers .Dense(712, activation= tf.nn.relu, kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .BatchNormalization())
        model.add(layers. Dropout(0.3))
        model.add(layers. Dense(512, activation= tf.nn.relu, kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers. BatchNormalization())
        model.add(layers. Dense(128, activation= tf.nn.relu, kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers. Dropout(0.4))

        model.add(layers. Dense(classes, activation= tf.nn.softmax))

        return model