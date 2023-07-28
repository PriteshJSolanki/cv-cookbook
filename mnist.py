"""
Convolutional Neural Networks

The purpose of this script is to build and train a simple CNN to classify images from the MNIST 
database which are 2D images (i.e grayscale).
"""

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

class MNISTClassifier:
    def __init__(self) -> None:
        self.model = None
        self.model_file = './models/cnn_mnist.weights.best.hdf5'
        self.input_shape = None
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

    def visualize_img(self, img, label='Image'):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        thresh = img.max()/2.5
        w,h,chan = img.shape
        for x in range(w):
            for y in range(h):
                ax.annotate(round(img[x][y],2), xy=[y,x],
                            horizontalalignment='center', 
                            verticalalignment='center',
                            color='white' if img[x][y]<thresh else 'black')
        ax.imshow(img, cmap='gray')
        ax.set_title(label)
        plt.show()
    
    def normalize(self):
        # Normalize - Divide by 255
        self.X_train = self.X_train.astype('float32')/255
        self.X_test = self.X_test.astype('float32')/255 

    def normalize_scaler(self):
        # Normalize - MinMaxScaler
        # This approach is more generic than dividing by 255, but takes 4x longer.
        scaler = MinMaxScaler()
        self.X_train_normalized = scaler.fit_transform(self.X_train.reshape(-1, 1))
        self.X_train = self.X_train_normalized.reshape(self.X_train.shape)
        self.X_test_normalized = scaler.fit_transform(self.X_test.reshape(-1, 1))
        self.X_test = self.X_test_normalized.reshape(self.X_test.shape)        

    def one_hot_encode(self):
        # One-Hot Encoding
        # Convert categorical labels to numerical. Each label gets a unique binary type value
        num_categories = len(np.unique(self.y_train))
        self.y_train = np_utils.to_categorical(self.y_train, num_categories)
        self.y_test = np_utils.to_categorical(self.y_test, num_categories)

    def reshape(self):
        # Reshape
        # To use a CNN, we must reshape the data to the following format:
        # input_shape = (height, width, channels)
        # training_data = (samples, height, width, channels)
        img_rows, img_cols = 28, 28  # MNIST data is 28x28
        channels = 1  # grayscale
        self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, channels)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, channels)
        self.input_shape = (img_rows, img_cols, 1)

    def preprocess(self):
        self.normalize()
        self.one_hot_encode()
        self.reshape()

    def build(self):
        ############################################################################################
        #  Hyperparameters
        ############################################################################################

        # Filters
        # The number of neurons in the hidden layer.
        # Ex: Total num of hidden units = (filter_height * filter_width * input_channels + 1) * output_channels
        #                               = (3*3*1 + 1) * 32
        # Greater number of units means more complex features can be detected, but at greater time and 
        # memory costs
        filters = 32

        # Kernel
        # Size of the convolution filter matrix
        # The window size that slides over the image and does some math to get the new "convolved" image 
        # for the next layer. Each kernel contains the weights that are multipled by each pixel in
        # the receptive field and summed together give the value of the center pixel in the new image.
        # Smaller windows capture more features, but take longer
        kernel_size = (3,3)

        # Strides
        # The amount the filter slides over the image. Greater numbers reduce output volume
        strides = (1,1)

        # Padding
        # Zero-padding adds zeros around the border of the image. 
        # Allows us to preserve the spatial size of the input volume. Helps build deeper networks that 
        # would normally shrink the height/width of the image as we get to deeper layers
        padding = 'same'

        # Pool Size
        # Pooling (Subsampling/Downsampling) reduces the size of the network by reducing the number of 
        # parameters passed to the next layer
        # Similar to kernel size, except there are no weights. They simply do a max or averaging within the
        # window
        # Output size will be previous layer size / pool_size
        pool_size = (2,2)

        ############################################################################################
        #  Build/Compile/Train
        ############################################################################################
        # Architecture
        # INPUT -> CONV_1 -> POOL_1 -> CONV_2 -> POOL_2 -> Flatten -> FC_1 -> FC_2
        self.model = Sequential()

        # CONV_1
        self.model.add(Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu', 
                        input_shape=self.input_shape))
        # POOL_1
        self.model.add(MaxPooling2D(pool_size))

        # CONV_2
        self.model.add(Conv2D(filters*2, kernel_size, padding=padding, activation='relu'))
        # POOL_1
        self.model.add(MaxPooling2D(pool_size))

        # flatten since too many dimensions and we want a classification output
        self.model.add(Flatten())

        # FC_1
        # Extracts classification data
        self.model.add(Dense(64, activation='relu'))

        # FC_2: Output layer
        # Classify the output into 10 classes (0-9)
        # Use softmax activation since we are classifying into 10 categories
        self.model.add(Dense(10, activation='softmax'))

        self.model.summary()

        # Compile
        # Loss function = Categorical Crossentropy since this is a classification problem
        # Optimizer = RMSProp is a popular gradient descent self.model (could have also used adam)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def train(self):
        # Train
        checkpoint = ModelCheckpoint(filepath=self.model_file, verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
        hist = self.model.fit(self.X_train, self.y_train, 
                        batch_size=32, 
                        epochs=12, 
                        validation_data=(self.X_test, self.y_test),
                        callbacks=[checkpoint, early_stop],
                        verbose=2,
                        shuffle=True)

    def evaluate(self):
        # evaluate test accuracy
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        accuracy = 100*score[1]
        print('Test accuracy: %.4f%%' % accuracy)
    
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
    model = MNISTClassifier()

    # Run the model end to end
    model.run()

    # Eval and tests can be run after loading the model
    # model.load()
    # model.evaluate()
    # model.test()