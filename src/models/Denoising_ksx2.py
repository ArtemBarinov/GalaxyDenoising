from keras.layers import Conv2D, BatchNormalization, Dropout, Conv2DTranspose, Add, Input, GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam



class Denoising_ksx2:
    def __init__(self, learn_rate=0.0001, kernel_size=64, filter_size=(3, 3), loss='mse'):
        '''
        Initialisation of the object that sets parameters for the model.

        Inputs:
        - lern_rate,    float:     Optional parameter that sets the learning rate of the optimiserself. Default is 0.0001.
        - loss,         str:       Optional parameter that specifies the loss function to use for training. Default is
                                   MSE, however also possible to use Binary Crossentropy (binary_crossentropy).
        - kernel_size,  int:       Optional parameter that sets the size of the kernel in the hidden layers. Default is 64.
        - filter_size,  tuple:     Optinal parameter that sets the filter size of the hidden layer. Default is (3x3).

        '''

        self.learn_rate = learn_rate
        self.loss = 'binary_crossentropy'
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.checkpoint_path = './Deionising_ks_ks'

    def create_model(self, shape, stddev=0.1, skip_shortcuts=False, deconvolutional=False, loss='mse'):

        '''

        Skip shortcuts: https://arxiv.org/pdf/1606.08921.pdf
        '''

        inputs = Input(shape=shape)
        inputs_noizy = inputs
        inputs_noizy = GaussianNoise(stddev)(inputs_noizy)
        x1 = Conv2D(self.kernel_size, self.filter_size, padding='same', activation='relu')(inputs_noizy)
        x2 = Conv2D(self.kernel_size, self.filter_size, padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)

        x3 = Conv2D(self.kernel_size, self.filter_size, padding='same', activation='relu')(x2)
        x4 = Conv2D(self.kernel_size, self.filter_size, padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = Dropout(0.2)(x4)

        if deconvolutional==False:
            x5 = Conv2D(self.kernel_size, self.filter_size, padding='same', activation='relu')(x4)
            x6 = Conv2D(self.kernel_size, self.filter_size, padding='same', activation='relu')(x5)
            x6 = BatchNormalization()(x6)
            x6 = Dropout(0.2)(x6)

            x7 = Conv2D(self.kernel_size, self.filter_size, padding='same', activation='relu')(x6)
            x8 = Conv2D(self.kernel_size, self.filter_size, padding='same', activation='relu')(x7)
            x8 = BatchNormalization()(x8)
            x8 = Dropout(0.2)(x8)

            outputs = Conv2D(3, self.filter_size, padding='same', activation='sigmoid')(x8)

        else :
            x5 = Conv2DTranspose(self.kernel_size, self.filter_size, padding='same', activation='relu')(x4)
            x6 = Conv2DTranspose(self.kernel_size, self.filter_size, padding='same', activation='relu')(x5)
            x6 = BatchNormalization()(x6)
            x6 = Dropout(0.2)(x6)

            x7 = Conv2DTranspose(self.kernel_size, self.filter_size, padding='same', activation='relu')(x6)
            x8 = Conv2DTranspose(self.kernel_size, self.filter_size, padding='same', activation='relu')(x7)
            x8 = BatchNormalization()(x8)
            x8 = Dropout(0.2)(x8)

            outputs = Conv2DTranspose(shape[-1], self.filter_size, padding='same', activation='sigmoid')(x8)

        if skip_shortcuts==True:
            y = BatchNormalization()(inputs)
            y = LeakyReLU()(y)
            outputs = Add([outputs, y])


        model = Model(inputs=inputs, outputs=outputs)

        model.summary()

        model.compile(loss=loss, optimizer=Adam(lr=self.learn_rate), metrics=['accuracy'])

        return model
