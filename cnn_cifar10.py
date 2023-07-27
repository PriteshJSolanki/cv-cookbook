"""
Convolutional Neural Networks

The purpose of this script is to build and train a CNN to classify images from the CIFAR-10 database
which are color images containing 1 of 10 object classes with 6k images per class.
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

class CNN:
    def __init__(self) -> None:
        self.model = None
        self.model_file = './models/cnn_cifar10.weights.best.hdf5'
        self.X_val = None  # Validation data
        self.y_val = None
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
        num_categories = len(np.unique(self.y_train))
        self.y_train = np_utils.to_categorical(self.y_train, num_categories)
        self.y_test = np_utils.to_categorical(self.y_test, num_categories)

    def preprocess(self):
        self.normalize()
        self.one_hot_encode()

        # Split data into training and validation
        self.X_train, self.X_val = (self.X_train[5000:], self.X_train[:5000])
        self.y_train, self.y_val = (self.y_train[5000:], self.y_train[:5000])

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
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        accuracy = 100*score[1]
        print('Test accuracy: %.4f%%' % accuracy)
    
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



if __name__ == '__main__':
    cnn = CNN()

    # Run the model end to end
    cnn.run()

    # Eval and tests can be run after loading the model
    # cnn.load()
    # cnn.evaluate()
    # cnn.test()