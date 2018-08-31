from keras.layers import Conv2D, BatchNormalization, Dropout, Conv2DTranspose, Add, Input, GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam



class Denoising_ksx6:
    '''
    Most of this needs more functional programming, as there is a lot of repitition.
    '''
    def __init__(self, learn_rate=0.0001, kernel_size=64, filter_size=(3, 3)):

        self.learn_rate = learn_rate
        self.loss = 'binary_crossentropy'
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.checkpoint_path = './Deionising_ks_ks'

    def create_model(self, shape, stddev=0.1, skip_shortcuts=False, kernel_size=None, deconvolutional=False, loss='mse', padding='same', kernel_initializer='he_normal'):

        '''

        Skip shortcuts: https://arxiv.org/pdf/1606.08921.pdf
        '''

        # If kernel size given here, change default kernel size.
        if kernel_size != None:
            self.kernel_size=kernel_size


        inputs = Input(shape=shape)
        inputs_noizy = inputs
        inputs_noizy = GaussianNoise(stddev)(inputs_noizy)
        x1 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(inputs_noizy)
        x2 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)

        # Shortcut 1 from here


        x3 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x2)
        x4 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x3)
        x4 = BatchNormalization()(x4)
        x4 = Dropout(0.2)(x4)

        x5 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x4)
        x6 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x5)
        x6 = BatchNormalization()(x6)
        x6 = Dropout(0.2)(x6)

        # Shortcut 2 from here

        x7 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x6)
        x8 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x7)
        x8 = BatchNormalization()(x8)
        x8 = Dropout(0.2)(x8)

        x9 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x8)
        x10 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x9)
        x10 = BatchNormalization()(x10)
        x10 = Dropout(0.2)(x10)

        # Shortcut 3 from here

        x11 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x10)
        x12 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x11)
        x12 = BatchNormalization()(x12)
        x12 = Dropout(0.2)(x12)



        if deconvolutional==False:
            x13 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x12)
            x14 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x13)
            x14 = BatchNormalization()(x14)
            x14 = Dropout(0.2)(x14)

            if skip_shortcuts==True:
                y = BatchNormalization()(x10)
                y = LeakyReLU()(y)
                x12 = Add([x14, y])

            x15 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x14)
            x16 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x15)
            x16 = BatchNormalization()(x16)
            x16 = Dropout(0.2)(x16)

            x17 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x16)
            x18 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x17)
            x18 = BatchNormalization()(x18)
            x18 = Dropout(0.2)(x18)

            if skip_shortcuts==True:
                y = BatchNormalization()(x6)
                y = LeakyReLU()(y)
                x12 = Add([x18, y])

            x19 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x18)
            x20 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x19)
            x20 = BatchNormalization()(x20)
            x20 = Dropout(0.2)(x20)

            x21 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x20)
            x22 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x21)
            x22 = BatchNormalization()(x22)
            x22 = Dropout(0.2)(x22)

            if skip_shortcuts==True:
                y = BatchNormalization()(x2)
                y = LeakyReLU()(y)
                x12 = Add([x22, y])

            x23 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x22)
            x24 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x23)
            x24 = BatchNormalization()(x24)
            x24 = Dropout(0.2)(x24)


            outputs = Conv2DTranspose(3, self.filter_size, padding='same', activation='sigmoid', kernel_initializer=kernel_initializer)(x24)

        else :

            x13 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x12)
            x14 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x13)
            x14 = BatchNormalization()(x14)
            x14 = Dropout(0.2)(x14)

            if skip_shortcuts==True:
                y = BatchNormalization()(x10)
                y = LeakyReLU()(y)
                x12 = Add([x14, y])

            x15 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x14)
            x16 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x15)
            x16 = BatchNormalization()(x16)
            x16 = Dropout(0.2)(x16)

            x17 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x16)
            x18 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x17)
            x18 = BatchNormalization()(x18)
            x18 = Dropout(0.2)(x18)

            if skip_shortcuts==True:
                y = BatchNormalization()(x6)
                y = LeakyReLU()(y)
                x12 = Add([x18, y])

            x19 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x18)
            x20 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x19)
            x20 = BatchNormalization()(x20)
            x20 = Dropout(0.2)(x20)

            x21 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x20)
            x22 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x21)
            x22 = BatchNormalization()(x22)
            x22 = Dropout(0.2)(x22)

            if skip_shortcuts==True:
                y = BatchNormalization()(x2)
                y = LeakyReLU()(y)
                x12 = Add([x22, y])

            x23 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x22)
            x24 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer=kernel_initializer)(x23)
            x24 = BatchNormalization()(x24)
            x24 = Dropout(0.2)(x24)


            outputs = Conv2DTranspose(shape[-1], self.filter_size, padding='same', activation='sigmoid', kernel_initializer=kernel_initializer)(x24)


        model = Model(inputs=inputs, outputs=outputs)

        model.summary()

        model.compile(loss=loss, optimizer=Adam(lr=self.learn_rate), metrics=['accuracy'])

        return model
