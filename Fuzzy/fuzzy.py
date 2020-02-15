import torch
import numpy as np

class Activations:

    """
    Regular activation functions used in the module are defined here.
    The activation functions used are:
        * Sigmoid
        * Relu
        * Swish
        * Leaky Relu
        * tanh
    """
    ## Sigmoid Activation
    @staticmethod
    def Sigmoid(z):
        z = 1/(1 + np.exp(-x))
        return z

    ## Relu Activation
    @staticmethod
    def Relu(z):
        z = max(0,z)
        return z

    ## Tanh Activation
    @staticmethod
    def Tanh(z):
        z = np.tanh(z)
        return z

    ## Leaky Relu Activation
    @staticmethod
    def Leaky_Relu(z):
        z = np.where(z > 0, z, z * 0.01)
        return z

    ## Swish Activation
    @staticmethod
    def Swish(z):
        z = z*Sigmoid(z)
        return z

    ## Functions for T Norm
    """
        Avialble T Norm Activations
            * Product T Norm
            * Minimum T Norm
            * Luckasiewickz T Norm
    """


    # Minimum T Norm
    @staticmethod
    def t_norm_min(mode,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            z = []
            for i in weight_matrix:
                z.append(min(i,input_matrix))
            return z
        elif mode == "v":
            return min(z)

    # Product T Norm
    @staticmethod
    def prod_t_norm(mode,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            z = torch.mm(weight_matrix, input_matrix)
            return z
        elif mode == "v":
            return torch.prod(z)

    # Luckasiewickz T Norm
    @staticmethod
    def luka_t_norm(mode,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            z = []
            for i in weight_matrix:
                z.append(torch.max(i + input_matrix - 1,0))
            return z
        elif mode == "v":
            z = torch.max(torch.sum(z)-1,0)
            return z


    ## Functions for T Co Norm or S Norm
    """
        Available T Co Norm or S Norm Activations
            * Probablistic Sum S Norm
            * Luckasiewickz S Norm
            * Maximum S Norm
    """

    # Maximum  S Norm
    def s_norm_max(mode,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            z = []
            for i in weight_matrix:
                z.append(torch.max(i, input_matrix))
        elif mode == "v":
            return torch.max(z)

    # Probablistic Sum S Norm
    def prob_s_norm(mode,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            z = []
            for i in weight_matrix:
                z.append((i + input_matrix) - i*weight_matrix)
            return z
        elif mode == "v":
            return (torch.sum(z, axis = 1) - torch.prod(z))

    # Luckasiewickz S Norm
    def luka_s_norm(mode,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            z = torch.empty(1,weight_matrix.size()[1])
            p = torch.empty(weight_matrix.size()[1])
            for i in weight_matrix:
                p = (torch.min(i + input_matrix.squeeze(1),torch.ones(i.size()[0])))
                """
                    Incomplete
                """
        elif mode == "v":
            z = torch.min(torch.sum(z, axis = 1), torch.ones(z.size()[0]))
            z = torch.view(z.size()[0],1)
            return z

class support:

    """This class includes all the support functions used in the package"""

    ## Function for normalising a vector
    @staticmethod
    def normalise(x):

        maxi = torch.max(x)
        mini = torch.min(x)

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
        self.activations = []

        ## Input Shape
        self.input_shape = input_shape
        ## Garbage Variables
        self.prev_layer_shape = 0


    def add_layer(self, no_of_neurons, layer_activation, t_norm = "min", s_norm = "max"):
        """
        no_of_neurons ==> Represent the Number of Neurons in that particular layer
        activation ==> Represent the activation to be used for that particular layer
        """
        ## Create the current layer
        layer = torch.empty(no_of_neurons, 1,device = self.device)

        if self.no_of_layers == 0:
            self.prev_layer_shape = self.input_shape

        ## Initialize weights between the current layer and previous layer
        weights = torch.rand(no_of_neurons, self.prev_layer_shape, device = self.device)

        ## Add the layer to the model architecture
        self.no_of_layers +=1
        self.layers.append(layer)
        self.prev_layer_shape = no_of_neurons

        ## Add the intiated weights to weights list
        self.weights.append(weights)

        ## Add the activation to the layer
        dict = {'act':layer_activation, 't_norm':t_norm, 's_norm':s_norm}
        self.activations.append(dict)


    def compute_layer(self, weight_matrix, input_matrix, layer_number):
        ## Device is GPU
        device = self.device

        if self.activations[layer_number]['act'] == 'AND':
            """
                AND Neuron ==> T(S(x,y))
            """
            if self.activations[layer_number]['s_norm'] == "max":
                z = Activations.s_norm_max("b",weight_matrix, input_matrix)
            elif self.activations[layer_number]['s_norm'] == "luka":
                z = Activations.luka_s_norm("b",weight_matrix, input_matrix)
            elif self.activations[layer_number]['s_norm'] == "prob":
                z = Activations.prob_s_norm("b",weight_matrix, input_matrix)

            if self.activations[layer_number]['t_norm'] == "min":
                z = Activations.t_norm_min("v",z)
            elif self.activations[layer_number]['t_norm'] == "luka":
                z = Activations.luka_t_norm("v",z)
            elif self.activations[layer_number]['t_norm'] == "prod":
                z = Activations.prod_t_norm("v",z)

        elif self.activations[layer_number]['act'] == 'OR':
            """
                OR Neuron ==> S(T(x,y))
            """
            if self.activations[layer_number]['t_norm'] == "min":
                z = Activations.t_norm_min("b",weight_matrix, input_matrix)
            elif self.activations[layer_number]['t_norm'] == "luka":
                z = Activations.luka_t_norm("b",weight_matrix, input_matrix)
            elif self.activations[layer_number]['t_norm'] == "prod":
                z = Activations.prod_t_norm("b",weight_matrix, input_matrix)

            if self.activations[layer_number]['s_norm'] == "max":
                z = Activations.s_norm_max("v",z)
            elif self.activations[layer_number]['s_norm'] == "luka":
                z = Activations.luka_s_norm("v",z)
            elif self.activations[layer_number]['s_norm'] == "prob":
                z = Activations.prob_s_norm("v",z)

        self.layers[layer_number] == z
        print(self.layers[layer_number].size())

    def train_model(self):
        for i in range(1, self.no_of_layers-1):
            Model.compute_layer(self,self.weights[i],self.layers[i-1],i)
        # print(self.layers[1].size())
