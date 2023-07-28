'''
Base class implementation of CNN

'''
import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

class CNN:
    def __init__(self) -> None:
        self.model = None
        self.model_file = './models/cnn.weights.best.hdf5'
        self.X_val = None  # Validation data
        self.y_val = None
        self.num_classes = None
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data() # Load dataset

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
        pass

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

if __name__ == '__main__':
    model = CNN()