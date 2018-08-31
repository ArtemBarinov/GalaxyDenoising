# Do all necessary imports
from keras.layers import Conv2D, BatchNormalization, Dropout, Conv2DTranspose, Add, Input, GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam


class DenoisingBaseline:
    '''
    A class that creates a instance of a denoising convolutional neural network that constists
    of 6 layers, including 4 hidden layers, and a regularisation layer that adds noise to the inputs
    image during training.
    '''
    def __init__(self, learn_rate=0.0001, loss='mse', kernel_size=64, filter_size=(3, 3)):
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
        self.loss = loss
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.checkpoint_path = './DenoisingBaseline'

    def create_model(self, shape, stddev=0.1, deconvolutional=False, padding='same'):
        '''
        Method to output a trainable model.

        Inputs:
        - shape, touple:            Shape of each image. Default used is channels last (120,120,3) for RGB
                                    and (120, 120, 1) for greyscale, corresponding to images of 120 by 120
                                    pixels in size.
        - stddev, double:           The standard deviation of the Gaussian Noise (GaussianNoise) to be added.
                                    Increase for a noisier image. Default is 0.1.
        - deconvolutional, Boolean: Specifies whether the model uses convolutional or deconvolutional layers
                                    for the decoder. Must use deconvolutional if padding='valid' (no padding)
                                    to have output size match input size.
        - padding, str:             Specifies whetehr to zero pad inputs of convolutional layer or not. 'same'
                                    results in the output being the same size as the output, 'valid' mean no
                                    padding.
        '''

        # Specify Input
        inputs = Input(shape=shape)

        # Add Noise. This workaround ensures that the correct input is passed
        # to the network for both training and testing.
        inputs_noizy = inputs
        inputs_noizy = GaussianNoise(stddev)(inputs_noizy) # Only active during training.

        # Encoder:
        x1 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(inputs_noizy)
        x2 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)

        # Decoder. If statement to differentiate between convolutional and deconvolutional layers.
        if deconvolutional==False:
            x3 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x2)
            x4 = Conv2D(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x3)
            x4 = BatchNormalization()(x4)
            x4 = Dropout(0.2)(x4)

        else:
            x3 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x2)
            x4 = Conv2DTranspose(self.kernel_size, self.filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x3)
            x4 = BatchNormalization()(x4)
            x4 = Dropout(0.2)(x4)

        # Output layer.
        outputs = Conv2D(shape[-1], self.filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x4)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Display the model summary, insluding structure and layer output sizes.
        model.summary()

        # Compile
        model.compile(loss=self.loss, optimizer=Adam(lr=self.learn_rate), metrics=['accuracy'])

        return model
