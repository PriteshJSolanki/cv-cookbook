"""
This script implements the AlexNet CNN design pattern

As this model was originally designed for training on the ImageNet dataset which contain larger image
files, this implementation adjusts the model to train on the CIFAR10 dataset

"""
from cnn import CNN
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense,Activation, MaxPooling2D, BatchNormalization, Dropout
from keras import regularizers, optimizers
from keras.regularizers import l2

class AlexNet(CNN):
    def __init__(self) -> None:
        super().__init__()
        self.model_file = './models/alexnet.weights.best.hdf5'

    def build(self):
        self.model = Sequential(name="Alexnet")

        # CONV1 + POOL1
        self.model.add(Conv2D(filters= 96, kernel_size= (3,3), strides=(1,1), padding='valid', kernel_regularizer=l2(0.0005),
        input_shape = self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides= (2,2), padding='valid'))
        self.model.add(BatchNormalization())
            
        # CONV2 + POOL2
        self.model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', kernel_regularizer=l2(0.0005)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        self.model.add(BatchNormalization())
                    
        # CONV3
        self.model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0005)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
                
        # CONV4
        self.model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0005)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
                    
        # CONV5
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0005)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        # FLATTEN (for input to FC layer)
        self.model.add(Flatten())

        # FC1
        self.model.add(Dense(units = 4096, activation = 'relu'))
        self.model.add(Dropout(0.5))

        # FC2
        self.model.add(Dense(units = 4096, activation = 'relu'))
        self.model.add(Dropout(0.5))
                                
        # FC3 (Output layer)
        self.model.add(Dense(units = self.num_classes, activation = 'softmax'))

        # print the self.model summary
        self.model.summary()

        # Optimizers 
        optimizer = optimizers.RMSprop(learning_rate=0.0003, decay=1e-6)

        # Compile
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

if __name__ == '__main__':
    model = AlexNet()
    
    # Run the model (build, train, evaluate)
    # model.run()

    # Eval and tests can be run after loading the model
    # model.load()
    # model.evaluate()
    # model.test()