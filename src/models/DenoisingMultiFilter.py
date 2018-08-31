from keras.layers import Conv2D, BatchNormalization, Dropout, Conv2DTranspose, Add, Input, GaussianNoise, add
from keras.models import Sequential, Model
from keras.optimizers import Adam


class DenoisingMultiFilter:
    def __init__(self, learn_rate=0.0001, loss='mse', kernel_size=64, filter_sizes=[(3, 3),(4,4)]):
        '''
        Initialisation of the object that sets parameters for the model.

        Inputs:
        - lern_rate,    float:              Optional parameter that sets the learning rate of the optimiserself. Default is 0.0001.
        - loss,         str:                Optional parameter that specifies the loss function to use for training. Default is
                                            MSE, however also possible to use Binary Crossentropy (binary_crossentropy).
        - kernel_size,  int:                Optional parameter that sets the size of the kernel in the hidden layers. Default is 64.
        - filter_size,  array of tuples:    Optinal parameter that sets the filter sizes of the hidden layers. Default is [(3x3), (4x4)],
                                            smalles filter sizes that could be reasonably used.

        '''
        self.learn_rate = learn_rate
        self.loss = loss
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.checkpoint_path = './DenoisingBaseline'


    def filterBranch(self, input, shape, kernel_size, filter_size, deconvolutional=False, padding='same'):
        '''
        This method creates one branch of the neural network.

        Inputs:

        - input, tensor:                    The input that the model recieves (batch of images).
        - shape, touple:                    Shape of each image. Default used is channels last (120,120,3) for RGB
                                            and (120, 120, 1) for greyscale, corresponding to images of 120 by 120
                                            pixels in size.
        - kernel_size, int:                 Parameter that sets the size of the kernel in the hidden layers.
                                            Increase for a noisier image. Default is 0.1.
        - filter_size,  array of tuples:    Parameter that sets the filter size of the hidden layer.
        - deconvolutional, Boolean:         Specifies whether the model uses convolutional or deconvolutional layers
                                            for the decoder. Must use deconvolutional if padding='valid' (no padding)
                                            to have output size match input size.
        - padding, str:                     Specifies whetehr to zero pad inputs of convolutional layer or not. 'same'
                                            results in the output being the same size as the output, 'valid' mean no
                                            padding.
        -
        '''

        # Encoder
        x1 = Conv2D(self.kernel_size, filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(input)
        x2 = Conv2D(self.kernel_size, filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)

        # Decoder
        if deconvolutional==False:
            x3 = Conv2D(self.kernel_size, filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x2)
            x4 = Conv2D(self.kernel_size, filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x3)
            x4 = BatchNormalization()(x4)
            x4 = Dropout(0.2)(x4)

        else:
            x3 = Conv2DTranspose(self.kernel_size, filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x2)
            x4 = Conv2DTranspose(self.kernel_size, filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x3)
            x4 = BatchNormalization()(x4)
            x4 = Dropout(0.2)(x4)

        # Output layer
        output = Conv2D(shape[-1], filter_size, padding=padding, activation='relu', kernel_initializer='he_normal')(x4)

        return output

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
        inputs_noizy = GaussianNoise(stddev)(inputs_noizy)

        # Use the fisrt specified filter size to create first branch.
        outputs = self.filterBranch(inputs_noizy, shape, self.kernel_size, self.filter_sizes[0], padding=padding, deconvolutional=deconvolutional)

        # Loop over the remaining filter sizes and sum their outputs.
        for filter_size in self.filter_sizes[1:]:
            branch = self.filterBranch(inputs_noizy, shape, self.kernel_size, filter_size, padding=padding, deconvolutional=deconvolutional)
            outputs = add([outputs, branch])



        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Display the model summary, insluding structure and layer output sizes.
        model.summary()

        # Compile
        model.compile(loss=self.loss, optimizer=Adam(lr=self.learn_rate), metrics=['accuracy'])

        return model
