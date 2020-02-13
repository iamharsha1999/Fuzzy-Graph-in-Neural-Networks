import torch
import numpy as np
from Activation import *

class support:

    """This class includes all the support functions used in the package"""

    ## Function for normalising a vector
    @staticmethod
    def normalise(x):

        maxi = np.max(x)
        mini = np.min(x)

        x = (x-mini)/(maxi-mini)

        return x

class Model:

    """Structuring the Neural Network Model"""


    @staticmethod
    def check_cuda(self):
        """Check if there is CUDA Device available for GPU accelaration"""

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Cuda Device Available")
            print("Name of the Cuda Device: ", torch.cuda.get_device_name())
            print("GPU Computational Capablity: ", torch.cuda.get_device_capability())

        else:

            print("Please check whether you have CUDA supported device")


    def __init__(self, input_shape, accelaration = "gpu",):

        self.accelaration = accelaration

        ## Define the parameters for a neural network model
        print("Initializing the Network...")

        ## Check for GPU Accelaration Availablity
        if self.accelaration  == "gpu":
            Model.check_cuda(self)

        self.layers = []
        self.no_of_layers = 0
        self.weights = []

        ## Input Shape
        self.input_shape = input_shape
        ## Garbage Variables
        self.prev_layer_shape = 0


    def add_layer(self, no_of_neurons):

        ## Create the current layer
        layer = torch.empty(no_of_neurons, device = self.device)

        if self.no_of_layers == 0:
            self.prev_layer_shape = self.input_shape

        ## Initialize weights between the current layer and previous layer
        weights = torch.rand(no_of_neurons, self.prev_layer_shape, device = self.device)

        ## Add the layer to the model architecture
        self.no_of_layers +=1
        self.layers.append(layer)

        ## Add the intiated weights to weights list
        self.weights.append(weights)

    def train_model():
