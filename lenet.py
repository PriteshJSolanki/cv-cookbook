"""
This script implements the LeNet CNN design pattern and will use the MNIST dataset to train

"""
from cnn import CNN
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class LeNet(CNN):
    def __init__(self) -> None:
        super().__init__(dataset='mnist')
        self.model_file = './models/lenet.weights.best.hdf5'
        self.epochs = 20

    def preprocess(self):
        self.normalize()
        self.one_hot_encode()
        self.reshape()

    def lr_schedule(self, epoch):
        # set the learning rate schedule as created in the original paper
        if epoch <= 2:     
            lr = 5e-4
        elif epoch > 2 and epoch <= 5:
            lr = 2e-4
        elif epoch > 5 and epoch <= 9:
            lr = 5e-5
        else: 
            lr = 1e-5
        return lr
    
    def build(self):
        self.model = Sequential(name="LeNet")

        # CONV1
        self.model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', 
                              input_shape=self.input_shape, padding='same'))
        # POOL1
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        # CONV2
        self.model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', 
                              padding='valid'))

        # S4 Pooling Layer
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

        # CONV3
        self.model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', 
                              padding='valid'))

        # FLATTEN
        self.model.add(Flatten())

        # FC1
        self.model.add(Dense(84, activation='tanh'))

        # FC2 (Output Layer)
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Summary
        self.model.summary()

        # Compile
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):
        # Train

        # Callbacks
        checkpoint = ModelCheckpoint(filepath=self.model_file, verbose=1, 
                                     save_best_only=True)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)  

        # train the model
        history = self.model.fit(self.X_train, self.y_train, 
                                    batch_size=self.batch_size, 
                                    epochs=self.epochs,
                                    validation_data=[self.X_test, self.y_test], 
                                    callbacks=[checkpoint, lr_scheduler], 
                                    verbose=2, shuffle=True)
        
        # plot learning curves of model losses
        df = pd.DataFrame(history.history)
        df['loss'].plot()
        df['val_loss'].plot()
        plt.legend()
        plt.show()

    def test(self, img_num:int = 0):
        # Make a prediction on a image
        test_image = np.expand_dims(self.X_test[img_num], axis=0)
        predictions = self.model.predict(test_image)
        for class_num, prob in enumerate(predictions[0]):
            print(f'Probability of class {class_num}: {prob*100}')
        predicted_class = np.argmax(predictions)
        actual_class = np.argmax(self.y_test[img_num])
        print(f'Predicted number is: {predicted_class}')
        print(f'Actual Number is: {actual_class}')

if __name__ == '__main__':
    model = LeNet()
    
    # Run the model (build, train, evaluate)
    # model.run()

    # Eval and tests can be run after loading the model
    # model.load()
    # model.evaluate()
    # model.test(10)