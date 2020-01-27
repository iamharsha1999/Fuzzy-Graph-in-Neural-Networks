import torch
import numpy as np

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


    def __init__(self, accelaration = "gpu"):

        self.accelaration = accelaration

        ## Define the parameters for a neural network model
        print("Initializing the Network...")

        ## Check for GPU Accelaration Availablity
        if self.accelaration  == "gpu":
            Model.check_cuda(self)

        self.layers = torch.empty(2, device = self.device)
        self.no_of_layers = 2
        self.weights = torch.empty(2, device = self.device)



        ## Garbage Variables
        self.prev_layer_shape = 0


    def add_layer(self, no_of_neurons, activation):

        ## Create the current layer
        layer = torch.empty(no_of_neurons)


        ## Initialize weights between the current layer and previous layer
        weights = torch.empty(no_of_neurons, self.prev_layer_shape)

        ## Add the layer to the model architecture
        self.no_of_layers +=1
        self.layers = torch.empty(no_of_layers)
