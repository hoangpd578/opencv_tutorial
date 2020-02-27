import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, models
from tensorflow.keras import callbacks, regularizers
class Vgg16:
    @staticmethod
    def build(width, hight, ndim, classes):
        weight_decay = 0.000

        model = models.Sequential()
        model.add(layers.Reshape(target_shape= [width, hight, ndim], input_shape= [width, hight, ndim]))

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
        model.add(layers .MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
        model.add(layers. BatchNormalization())
        model.add(layers .Dropout(0.3))

        #block 3
        model.add(layers .Conv2D(256, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .BatchNormalization())
        model.add(layers .Conv2D(256, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
        model.add(layers .BatchNormalization())
        model.add(layers. Conv2D(256, (3, 3), activation= tf.nn.relu,
                                 kernel_regularizer= regularizers.l2(weight_decay)))
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


model = Vgg16.build(224, 224, 3, 7)
model.summary()

optimizer = optimizers.Nadam(lr= 0.01)

model.compile(optimizer= optimizer, loss= tf.losses.SparseCategoricalCrossentropy, matrics= ["accuracy"])
early_stopping = callbacks.EarlyStopping(patience= 15)
learning_rate_reduce = callbacks.ReduceLROnPlateau(factor= 0.5, min_lr= 0.00001, patience= 5)
checkpont = callbacks.ModelCheckpoint("drive/My Drive/checkpoint.h5", save_best_only= True)

call_back = [early_stopping, learning_rate_reduce, checkpont]

history = model.fit(train_X, train_y, epochs= 150, batch_size= 128, callbacks= call_back, validation_split= 0.15)