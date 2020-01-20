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
    def check_cuda():
        """Check if there is CUDA Device available for GPU accelaration"""

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Cuda Device Available")
            print("Name of the Cuda Device: ", torch.cuda.get_device_name())

        else:

            print("Please check whether you have CUDA supported device")






    def __init__(self,x,y, accelaration = "gpu"):

        ## Define the parameters for a neural network model
        """
        x = input_data
        y = output_data
        """
        print("Initializing the Network...")
        x = torch.tensor(x)
        y = torch.tensor(y)
        print("Shape of the Input: " + x.size())
        print("Shape of the Output: " + y.size())

        ## Check for GPU Accelaration Availablity


        ## Initialise the model by creating the layers
        self.layers = torch.empty(2)
        self.no_of_layers = 2
        self.weights = torch.empty(2)
        self.accelaration = "gpu"

        if self.accelaration  = "gpu":
            check_cuda()

    def add_layer(no_of_neurons, activation):

        ## Create the current layer
        layer = torch.empty(no_of_neurons)

        ## Initialize weights between the current layer and previous layer
        weights = torch.empty

        ## Add the layer to the model architecture
        self.no_of_layers +=1
        self.layers = torch.empty(no_of_layers)
