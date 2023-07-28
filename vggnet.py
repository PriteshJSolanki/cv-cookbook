"""
This script implements the VGGNet CNN design pattern

"""
from cnn import CNN
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation, MaxPooling2D, Dropout
from keras import regularizers, optimizers
from keras.regularizers import l2

class VGGNet(CNN):
    def __init__(self) -> None:
        super().__init__()
        self.model_file = './models/vggnet.weights.best.hdf5'

    def build(self):
        self.model = Sequential(name="vggnet")

        # CONV1
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same',input_shape=self.input_shape))
        
        # CONV2
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # CONV3
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        # CONV4
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # CONV5
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        # CONV6
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        # CONV7
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # CONV8
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        # CONV9
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        # CONV10
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # CONV11
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        # CONV12
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        # CONV3
        self.model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', 
                              padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # FLATTEN
        self.model.add(Flatten())

        # FC1
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))

        # FC2
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))

        # FC3 (Output)
        self.model.add(Dense(1000, activation='softmax'))

        self.model.summary()

        # Optimizers 
        optimizer = optimizers.RMSprop(learning_rate=0.0003, decay=1e-6)

        # Compile
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

if __name__ == '__main__':
    model = VGGNet()
    
    # Run the model (build, train, evaluate)
    model.run()

    # Eval and tests can be run after loading the model
    # model.load()
    # model.evaluate()
    # model.test()