'''
Base class implementation of CNN

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

class CNN(ABC):
    def __init__(self, dataset='cifar') -> None:
        # Load dataset
        if dataset == 'cifar':
            (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
            # define text labels (source: https://www.cs.toronto.edu/~kriz/cifar.html)
            self.cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
                                   'horse', 'ship', 'truck']
        elif dataset == 'mnist':
            (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        
        self.X_val = None  # Validation data
        self.y_val = None
        self.model = None
        self.model_file = './models/cnn.weights.best.hdf5'
        self.num_classes = len(np.unique(self.y_train))
        
        # Model Hyper parameters
        self.input_shape=self.X_train[0].shape
        self.batch_size = 32
        self.epochs = 100

    def normalize(self):
        mean = np.mean(self.X_train)
        std = np.std(self.X_train)
        self.X_train = (self.X_train-mean)/(std+1e-7)
        self.X_test = (self.X_test-mean)/(std+1e-7) 

    def one_hot_encode(self):
        # One-Hot Encoding
        # Convert categorical labels to numerical. Each label gets a unique binary type value
        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

    def split_data(self):
        # Split data into training and validation
        self.X_train, self.X_val = (self.X_train[5000:], self.X_train[:5000])
        self.y_train, self.y_val = (self.y_train[5000:], self.y_train[:5000])

    def reshape(self):
        # Reshape the data to support mnist dataset
        img_rows, img_cols = self.X_train[0].shape

        self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, 1)
        self.input_shape = (img_rows, img_cols, 1)

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
        self.split_data()

    @abstractmethod
    def build(self):
        pass

    def train(self):
        # Train
        checkpoint = ModelCheckpoint(filepath=self.model_file, verbose=1, 
                                     save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)  
        hist = self.model.fit(self.X_train, self.y_train, 
                                batch_size=self.batch_size, 
                                epochs=self.epochs, 
                                validation_data=(self.X_test, self.y_test),
                                callbacks=[checkpoint, early_stop],
                                verbose=2,
                                shuffle=True)
        
        # plot learning curves of model losses
        df = pd.DataFrame(history.history)
        df['loss'].plot()
        df['val_loss'].plot()
        plt.legend()
        plt.show()

    def evaluate(self):
        # evaluate test accuracy
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print('\nAccuracy: %.3f Loss: %.3f' % (scores[1]*100, scores[0]))
    
    def test(self, img_num:int = 0):
        pass

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
        self.test()      

if __name__ == '__main__':
    model = CNN()