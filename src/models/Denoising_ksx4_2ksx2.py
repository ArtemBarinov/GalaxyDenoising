from keras.layers import Conv2D, BatchNormalization, Dropout, Conv2DTranspose, Add, Input, GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam



class Denoising_ksx4_2ksx2:
    '''
    Most of this needs more functional programming, as there is a lot of repitition.
    '''
    def __init__(self, learn_rate=0.0001, kernel_size=64, filter_size=(3, 3), loss='mse'):

        self.learn_rate = learn_rate
        self.loss = 'binary_crossentropy'
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.checkpoint_path = './Deionising_ks_ks'

    def create_model(self, shape, stddev=0.1, skip_shortcuts=False, kernel_size=None, deconvolutional=False, loss='binary_crossentropy', padding='same'):

        '''

        Skip shortcuts: https://arxiv.org/pdf/1606.08921.pdf
        '''

        # If kernel size given here, change default kernel size.
        if kernel_size != None:
            self.kernel_size=kernel_size


        inputs = Input(shape=shape)
        inputs_noizy = inputs
        inputs_noizy = GaussianNoise(stddev)(inputs_noizy)
        x1 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(inputs_noizy)
        x2 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)

        # Shortcut 1 from here


        x3 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x2)
        x4 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = Dropout(0.2)(x4)

        x5 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x4)
        x6 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x5)
        x6 = BatchNormalization()(x6)
        x6 = Dropout(0.2)(x6)

        # Shortcut 2 from here

        x7 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x6)
        x8 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x7)
        x8 = BatchNormalization()(x8)
        x8 = Dropout(0.2)(x8)

        x9 = Conv2D(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x8)
        x10 = Conv2D(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x9)
        x10 = BatchNormalization()(x10)
        x10 = Dropout(0.2)(x10)

        # Shortcut 3 from here

        x11 = Conv2D(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x10)
        x12 = Conv2D(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x11)
        x12 = BatchNormalization()(x12)
        x12 = Dropout(0.2)(x12)



        if deconvolutional==False:
            x13 = Conv2D(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x12)
            x14 = Conv2D(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x13)
            x14 = BatchNormalization()(x14)
            x14 = Dropout(0.2)(x14)

            if skip_shortcuts==True:
                y = BatchNormalization()(x10)
                y = LeakyReLU()(y)
                x12 = Add([x14, y])

            x15 = Conv2D(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x14)
            x16 = Conv2D(int(self.kernel_size*2).filter_size, padding=padding, activation='relu')(x15)
            x16 = BatchNormalization()(x16)
            x16 = Dropout(0.2)(x16)

            x17 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x16)
            x18 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x17)
            x18 = BatchNormalization()(x18)
            x18 = Dropout(0.2)(x18)

            if skip_shortcuts==True:
                y = BatchNormalization()(x6)
                y = LeakyReLU()(y)
                x12 = Add([x18, y])

            x19 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x18)
            x20 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x19)
            x20 = BatchNormalization()(x20)
            x20 = Dropout(0.2)(x20)

            x21 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x20)
            x22 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x21)
            x22 = BatchNormalization()(x22)
            x22 = Dropout(0.2)(x22)

            if skip_shortcuts==True:
                y = BatchNormalization()(x2)
                y = LeakyReLU()(y)
                x12 = Add([x22, y])

            x23 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x22)
            x24 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x23)
            x24 = BatchNormalization()(x24)
            x24 = Dropout(0.2)(x24)


            outputs = Conv2DTranspose(3, self.filter_size, padding='same', activation='sigmoid')(x24)

        else :

            x13 = Conv2DTranspose(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x12)
            x14 = Conv2DTranspose(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x13)
            x14 = BatchNormalization()(x14)
            x14 = Dropout(0.2)(x14)

            if skip_shortcuts==True:
                y = BatchNormalization()(x10)
                y = LeakyReLU()(y)
                x12 = Add([x14, y])

            x15 = Conv2DTranspose(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x14)
            x16 = Conv2DTranspose(int(self.kernel_size*2), self.filter_size, padding=padding, activation='relu')(x15)
            x16 = BatchNormalization()(x16)
            x16 = Dropout(0.2)(x16)

            x17 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x16)
            x18 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x17)
            x18 = BatchNormalization()(x18)
            x18 = Dropout(0.2)(x18)

            if skip_shortcuts==True:
                y = BatchNormalization()(x6)
                y = LeakyReLU()(y)
                x12 = Add([x18, y])

            x19 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x18)
            x20 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x19)
            x20 = BatchNormalization()(x20)
            x20 = Dropout(0.2)(x20)

            x21 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x20)
            x22 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x21)
            x22 = BatchNormalization()(x22)
            x22 = Dropout(0.2)(x22)

            if skip_shortcuts==True:
                y = BatchNormalization()(x2)
                y = LeakyReLU()(y)
                x12 = Add([x22, y])

            x23 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x22)
            x24 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu')(x23)
            x24 = BatchNormalization()(x24)
            x24 = Dropout(0.2)(x24)


            outputs = Conv2DTranspose(shape[-1], self.filter_size, padding='same', activation='sigmoid')(x24)


        model = Model(inputs=inputs, outputs=outputs)

        model.summary()

        model.compile(loss=loss, optimizer=Adam(lr=self.learn_rate), metrics=['accuracy'])

        return model
