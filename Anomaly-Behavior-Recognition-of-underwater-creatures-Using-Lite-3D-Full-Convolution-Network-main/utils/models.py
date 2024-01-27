from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, ReLU, Softmax, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from utils.focal_loss import *


def threeDCNN(inputShape, nb_classes):
    model = Sequential()

    model.add(Conv3D(16, kernel_size=(3, 3, 3), input_shape=inputShape, padding='same'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


def c3d(inputShape, nb_classes):
    model = Sequential()

    model.add(Conv3D(16, kernel_size=(3, 3, 3), input_shape=inputShape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1), strides=(1, 1, 1), padding='same'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


def lite3D(inputShape, nb_classes):
    model = Sequential()

    model.add(Conv3D(16, kernel_size=(3, 3, 1), input_shape=inputShape, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 1), padding='same'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='valid'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 1), padding='same'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv3D(nb_classes, kernel_size=(3, 3, 3), padding='valid'))
    model.add(Activation('softmax'))
    model.add(Flatten())
    return model


def model_comp(model, loss_function="cross", alpha=None):
    if loss_function == "focal":
        if alpha is None:
            alpha = [[1 - 0.0048], [1 - 0.3816], [1 - 0.1211], [1 - 0.1637], [1 - 0.0059], [1 - 0.4659], [1 - 0.0044]]
        model.compile(loss=[multi_category_focal_loss3(alpha=alpha, gamma=0)], optimizer=Adam(), metrics=['accuracy'])
    else:
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model


def getModel(args, inputShape, nb_classes, alpha=None):
    if args.model == "lite3d":
        model = lite3D(inputShape, nb_classes)
    elif args.model == "3dcnn":
        model = threeDCNN(inputShape, nb_classes)
    elif args.model == "c3d":
        model = c3d(inputShape, nb_classes)
    else:
        print("Pleas use current model.")

    if args.loss == "cross":
        model = model_comp(model)
    elif args.loss == "focal":
        model = model_comp(model, loss_function="focal", alpha=alpha)
    return model
