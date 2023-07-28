"""
Convolutional Neural Networks

The purpose of this script is to build and train a CNN to classify images from the CIFAR-10 database
which are color images containing 1 of 10 object classes with 6k images per class.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras import regularizers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

class CIFAR10Classifier:
    def __init__(self) -> None:
        self.model = None
        self.model_file = './models/cifar10classifier.weights.best.hdf5'
        self.X_val = None  # Validation data
        self.y_val = None
        self.num_classes = None
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data() # Load dataset

        # define text labels (source: https://www.cs.toronto.edu/~kriz/cifar.html)
        self.cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
                          'ship', 'truck']

    def visualize_img(img, label=''):
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        ax.set_title(label)
        ax.imshow(np.squeeze(img))
        plt.show()

    def normalize(self):
        # Normalize - Divide by 255
        self.X_train = self.X_train.astype('float32')/255
        self.X_test = self.X_test.astype('float32')/255    

    def one_hot_encode(self):
        # One-Hot Encoding
        # Convert categorical labels to numerical. Each label gets a unique binary type value
        self.num_classes = len(np.unique(self.y_train))
        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

    def split_data(self):
        # Split data into training and validation
        self.X_train, self.X_val = (self.X_train[5000:], self.X_train[:5000])
        self.y_train, self.y_val = (self.y_train[5000:], self.y_train[:5000])

    def preprocess(self):
        self.normalize()
        self.one_hot_encode()
        self.split_data()

    def build(self):
        ############################################################################################
        #  Hyperparameters
        ############################################################################################
        # Filters
        filters = 16

        # Kernel
        kernel_size = (2,2)

        # Strides
        strides = (1,1)

        # Padding
        padding = 'same'

        # Pool Size
        pool_size = (2,2)

        ############################################################################################
        #  Model
        ############################################################################################
        # Architecture
        # INPUT -> CONV_1 -> POOL_1 -> CONV_2 -> POOL_2 -> CONV_3 -> POOL_3 ->
        # DROPOUT -> FLATTEN -> FC_1 -> DROPOUT-> FC_2

        self.model = Sequential()

        # CONV/POOL_1
        self.model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, 
                         activation='relu',
                         input_shape=(32,32,3)))
        self.model.add(MaxPooling2D(pool_size))

        # CONV/POOL_2
        self.model.add(Conv2D(filters=filters*2, kernel_size=kernel_size, padding=padding, 
                         activation='relu',
                         input_shape=(32,32,3)))
        self.model.add(MaxPooling2D(pool_size))

        # CONV/POOL_3
        self.model.add(Conv2D(filters=filters*4, kernel_size=kernel_size, padding=padding, 
                         activation='relu',
                         input_shape=(32,32,3)))
        self.model.add(MaxPooling2D(pool_size))

        # DROPOUT
        self.model.add(Dropout(0.3))  # 30% dropout rate
        # FLATTEN
        self.model.add(Flatten())

        # FC_1
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dropout(0.4))  # 40% dropout rate
        # FC_2 (Output Layer)
        self.model.add(Dense(10, activation='softmax'))

        self.model.summary()

        # Compile
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        # Train
        checkpoint = ModelCheckpoint(filepath='./models/cnn_cifar10.weights.best.hdf5', verbose=1, 
                                     save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)  
        hist = self.model.fit(self.X_train, self.y_train,
                batch_size=32,
                epochs=100,
                validation_data=[self.X_val, self.y_val],
                callbacks=[checkpoint, early_stop],
                verbose=2, shuffle=True)

    def evaluate(self):
        # evaluate test accuracy
        scores = self.model.evaluate(self.X_test, self.y_test, batch_size=128, verbose=1)
        print('\nAccuracy: %.3f Loss: %.3f' % (scores[1]*100, scores[0]))
    
    def test(self, img_num:int = 0):
        # Make a prediction on 32 images and visualize the result
        predictions = self.model.predict(self.X_test)
        fig = plt.figure(figsize=(20,8))
        samples = np.random.choice(self.X_test.shape[0], size=32, replace=False)
        for i, sample_num in enumerate(samples):
            ax = fig.add_subplot(4,8, i+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(self.X_test[sample_num]))
            pred = np.argmax(predictions[sample_num])
            actual = np.argmax(self.y_test[sample_num])
            ax.set_title("{} ({})".format(self.cifar10_labels[pred], self.cifar10_labels[actual]),
                        color=("green" if pred == actual else "red"))
        plt.show()

    def load(self):
        # Run evertying but training
        self.preprocess()
        self.build()
        self.model.load_weights(self.model_file)

    def run(self):
        # Run the entire process
        self.preprocess()
        self.build()
        self.train()
        self.evaluate()        

class CIFAR10ClassifierPlus(CIFAR10Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.model_file = './models/cifar10classifierplus.weights.best.hdf5'
        self.datagen = None

    def normalize(self):
        mean = np.mean(self.X_train)
        std = np.std(self.X_train)
        self.X_train = (self.X_train-mean)/(std+1e-7)
        self.X_test = (self.X_test-mean)/(std+1e-7)  
    
    def augment_data(self):
        self.datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
            )

        # compute the data augmentation on the training set
        self.datagen.fit(self.X_train) 
    
    def preprocess(self):
        self.normalize()
        self.one_hot_encode()
        self.augment_data()

    def build(self):
        ############################################################################################
        #  Hyperparameters
        ############################################################################################        
        # number of hidden units variable 
        base_hidden_units = 32

        # l2 regularization hyperparameter
        weight_decay = 1e-4 

        ############################################################################################
        #  Model
        ############################################################################################
        self.model = Sequential()

        # CONV1
        self.model.add(Conv2D(base_hidden_units, (3,3), padding='same', 
                         kernel_regularizer=regularizers.l2(weight_decay), 
                         input_shape=self.X_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        # CONV2
        self.model.add(Conv2D(base_hidden_units, (3,3), padding='same', 
                         kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))

        # CONV3
        self.model.add(Conv2D(2*base_hidden_units, (3,3), padding='same', 
                         kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        # CONV4
        self.model.add(Conv2D(2*base_hidden_units, (3,3), padding='same', 
                         kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.3))

        # CONV5
        self.model.add(Conv2D(4*base_hidden_units, (3,3), padding='same', 
                         kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())

        # CONV6
        self.model.add(Conv2D(4*base_hidden_units, (3,3), padding='same', 
                         kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.4))

        # FC7
        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # print self.model summary
        self.model.summary()  

        # Optimizers
        # optimizer = keras.optimizers.RMSprop(learning_rate=0.001,decay=1e-6)
        # optimizer = optimizers.Adam(learning_rate=0.0005,decay=1e-6)   
        optimizer = optimizers.RMSprop(learning_rate=0.0003, decay=1e-6)

        # Compile
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train(self):
        batch_size = 64
        epochs=125

        checkpointer = ModelCheckpoint(filepath=self.model_file, verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25) 
        
        history = self.model.fit_generator(self.datagen.flow(self.X_train, self.y_train, batch_size=batch_size), 
                                      callbacks=[checkpointer, early_stop],
                                      steps_per_epoch=self.X_train.shape[0] // batch_size, 
                                      epochs=epochs,
                                      verbose=2,
                                      validation_data=(self.X_test,self.y_test))
        
        # plot learning curves of model losses
        df = pd.DataFrame(history.history)
        df['loss'].plot()
        df['val_loss'].plot()
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # model = CIFAR10Classifier()
    model = CIFAR10ClassifierPlus()
    
    # Run the model end to end
    # model.run()

    # Eval and tests can be run after loading the model
    model.load()
    model.evaluate()
    model.test()