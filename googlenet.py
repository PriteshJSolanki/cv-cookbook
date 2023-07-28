"""
This script implements the VGGNet CNN design pattern

"""
from cnn import CNN
import keras
import math
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,Dropout, Dense, Input, concatenate,\
                         AveragePooling2D,Flatten, MaxPooling2D
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd

class GoogLeNet(CNN):
    def __init__(self) -> None:
        super().__init__(dataset='cifar10')
        self.model_file = './models/googlenet.weights.best.hdf5'
        self.kernel_init = keras.initializers.glorot_uniform()
        self.bias_init = keras.initializers.Constant(value=0.2)
        self.epochs = 25
        self.initial_lrate = 0.01

    def preprocess(self):
        self.resize(224,224,1_000)
        self.normalize()
        self.one_hot_encode()

    def decay(self, steps=100):
        self.initial_lrate = 0.01
        drop = 0.96
        epochs_drop = 8
        lrate = self.initial_lrate * math.pow(drop, math.floor((1+self.epochs)/epochs_drop))
        return lrate

    def inception_module(self,
                         x,
                         filters_1x1,
                         filters_3x3_reduce,
                         filters_3x3,
                         filters_5x5_reduce,
                         filters_5x5,
                         filters_pool_proj,
                         name=None):

        conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', 
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        
        conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', 
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        
        conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', 
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_3x3)

        conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', 
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        
        conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', 
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_5x5)

        pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', 
                           kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(pool_proj)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
        
        return output
    
    def build(self):
        input_layer = Input(shape=(224, 224, 3))

        x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', 
                               name='conv_1_7x7/2', kernel_initializer=self.kernel_init, 
                               bias_initializer=self.bias_init)(input_layer)
        x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
        x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
        x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

        x = self.inception_module(x,
                            filters_1x1=64,
                            filters_3x3_reduce=96,
                            filters_3x3=128,
                            filters_5x5_reduce=16,
                            filters_5x5=32,
                            filters_pool_proj=32,
                            name='inception_3a')

        x = self.inception_module(x,
                            filters_1x1=128,
                            filters_3x3_reduce=128,
                            filters_3x3=192,
                            filters_5x5_reduce=32,
                            filters_5x5=96,
                            filters_pool_proj=64,
                            name='inception_3b')

        x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

        x = self.inception_module(x,
                            filters_1x1=192,
                            filters_3x3_reduce=96,
                            filters_3x3=208,
                            filters_5x5_reduce=16,
                            filters_5x5=48,
                            filters_pool_proj=64,
                            name='inception_4a')


        classifier_1 = AveragePooling2D((5, 5), strides=3)(x)
        classifier_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(classifier_1)
        classifier_1 = Flatten()(classifier_1)
        classifier_1 = Dense(1024, activation='relu')(classifier_1)
        classifier_1 = Dropout(0.7)(classifier_1)
        classifier_1 = Dense(10, activation='softmax', name='auxilliary_output_1')(classifier_1)

        x = self.inception_module(x,
                            filters_1x1=160,
                            filters_3x3_reduce=112,
                            filters_3x3=224,
                            filters_5x5_reduce=24,
                            filters_5x5=64,
                            filters_pool_proj=64,
                            name='inception_4b')

        x = self.inception_module(x,
                            filters_1x1=128,
                            filters_3x3_reduce=128,
                            filters_3x3=256,
                            filters_5x5_reduce=24,
                            filters_5x5=64,
                            filters_pool_proj=64,
                            name='inception_4c')

        x = self.inception_module(x,
                            filters_1x1=112,
                            filters_3x3_reduce=144,
                            filters_3x3=288,
                            filters_5x5_reduce=32,
                            filters_5x5=64,
                            filters_pool_proj=64,
                            name='inception_4d')


        classifier_2 = AveragePooling2D((5, 5), strides=3)(x)
        classifier_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(classifier_2)
        classifier_2 = Flatten()(classifier_2)
        classifier_2 = Dense(1024, activation='relu')(classifier_2)
        classifier_2 = Dropout(0.7)(classifier_2)
        classifier_2 = Dense(10, activation='softmax', name='auxilliary_output_2')(classifier_2)

        x = self.inception_module(x,
                            filters_1x1=256,
                            filters_3x3_reduce=160,
                            filters_3x3=320,
                            filters_5x5_reduce=32,
                            filters_5x5=128,
                            filters_pool_proj=128,
                            name='inception_4e')

        x = AveragePooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

        x = self.inception_module(x,
                            filters_1x1=256,
                            filters_3x3_reduce=160,
                            filters_3x3=320,
                            filters_5x5_reduce=32,
                            filters_5x5=128,
                            filters_pool_proj=128,
                            name='inception_5a')

        x = self.inception_module(x,
                            filters_1x1=384,
                            filters_3x3_reduce=192,
                            filters_3x3=384,
                            filters_5x5_reduce=48,
                            filters_5x5=128,
                            filters_pool_proj=128,
                            name='inception_5b')

        x = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid',name='avg_pool_5_3x3/1')(x)

        x = Dropout(0.4)(x)
        x = Dense(1000, activation='relu', name='linear')(x)
        x = Dense(self.num_classes, activation='softmax', name='output')(x)

        # Build the model
        self.model = Model(input_layer, [x, classifier_1, classifier_2], name='googlenet')
        self.model.summary()

        # Optimizer
        sgd = SGD(learning_rate=self.initial_lrate, momentum=0.9, nesterov=False)

        # Compile
        self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], 
                           loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])

    def train(self):
        checkpoint = ModelCheckpoint(filepath=self.model_file, verbose=1, 
                                     save_best_only=True)
        lr_sc = LearningRateScheduler(self.decay, verbose=1)
        
        # Fit
        y_train_combined = [self.y_train, self.y_train, self.y_train]
        y_test_combined = [self.y_test, self.y_test, self.y_test]
        history = self.model.fit(self.X_train, y_train_combined, 
                                 validation_data=(self.X_test, y_test_combined), 
                                 epochs=self.epochs, batch_size=256, callbacks=[lr_sc, checkpoint])
        
        # plot learning curves of model losses
        df = pd.DataFrame(history.history)
        df['loss'].plot()
        df['val_loss'].plot()
        plt.legend()
        plt.show()

if __name__ == '__main__':
    model = GoogLeNet()

    # Run the model (build, train, evaluate)
    model.run()

    # Eval and tests can be run after loading the model
    # model.load()
    # model.evaluate()
    # model.test()