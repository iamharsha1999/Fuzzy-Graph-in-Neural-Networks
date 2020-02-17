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
    def t_norm_min(mode,device,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            a = torch.randn(weight_matrix.size()[0], weight_matrix.size()[1], device = device, requires_grad = True)
            j = 0
            for i in weight_matrix:
                a[j] = torch.min(i,input_matrix.squeeze(1))
                j+=1
            return a
        elif mode == "v":

            z = torch.min(z, axis =1).values
            z = z.view(-1,1)
            return z

    # Product T Norm
    @staticmethod
    def prod_t_norm(mode,device,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            a = torch.randn(weight_matrix.size()[0], weight_matrix.size()[1], device = device, requires_grad = True)
            j = 0
            for i in weight_matrix:
                a[j] = torch.mul(i, input_matrix.squeeze(1))
                j+=1
            return a
        elif mode == "v":
            z = torch.prod(z, axis = 1)
            z = z.view(-1,1)
            return z

    # Luckasiewickz T Norm
    @staticmethod
    def luka_t_norm(mode, device, weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            a = torch.rand(weight_matrix.size()[0], weight_matrix.size()[1], device =device, requires_grad = True)
            j=0
            for i in weight_matrix:
                a[j] = torch.max(i + input_matrix.squeeze(1) - torch.ones(weight_matrix.size()[1], device = device),torch.zeros(weight_matrix.size()[1], device = device))
                j+=1
            b =  torch.empty(len(a), weight_matrix.size()[1], device = device)
            torch.cat(a,out =b)
            return a
        elif mode == "v":
            z = torch.max(torch.sum(z, axis =1)-torch.ones(z.size()[0], device = device),torch.zeros(z.size()[0], device = device))
            z = z.view(-1,1)
            return z


    ## Functions for T Co Norm or S Norm
    """
        Available T Co Norm or S Norm Activations
            * Probablistic Sum S Norm
            * Luckasiewickz S Norm
            * Maximum S Norm
    """

    # Maximum  S Norm
    def s_norm_max(mode,device,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            a = torch.rand(weight_matrix.size()[0], weight_matrix.size()[1], device = device, requires_grad = True)
            j=0
            for i in weight_matrix:
                a[j] = torch.max(i, input_matrix.squeeze(1))
                j+=1
            return a
        elif mode == "v":
            z = torch.max(z, axis = 1).values
            z = z.view(-1,1)
            return z
    # Probablistic Sum S Norm
    def prob_s_norm(mode,device,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            a = torch.rand(weight_matrix.size()[0], weight_matrix.size()[1], device =device, requires_grad = True)
            j=0
            for i in weight_matrix:
                a[j] = (i + input_matrix.squeeze(1)) - (i*input_matrix.squeeze(1))
                j+=1
            return a
        elif mode == "v":
            z = torch.sum(z, axis = 1) - torch.prod(z, axis = 1)
            z = z.view(-1,1)
            return z

    # Luckasiewickz S Norm
    def luka_s_norm(mode,device,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            a = torch.rand(weight_matrix.size()[0], weight_matrix.size()[1], device =device, requires_grad = True)
            j=0
            for i in weight_matrix:
                a[j] = torch.min(i + input_matrix.squeeze(1),torch.ones(i.size()[0], device = device))
                j+=1
            return a
        elif mode == "v":
            z = torch.min(torch.sum(z, axis = 1), torch.ones(z.size()[0]))
            z = z.view(-1,1)
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


    def add_layer(self, no_of_neurons, layer_activation, t_norm = "luka", s_norm = "luka"):
        """
        no_of_neurons ==> Represent the Number of Neurons in that particular layer
        activation ==> Represent the activation to be used for that particular layer
        """
        ## Create the current layer
        layer = torch.empty(no_of_neurons, 1,device = self.device, requires_grad = True)

        if self.no_of_layers == 0:
            self.prev_layer_shape = self.input_shape

        ## Initialize weights between the current layer and previous layer
        weights = torch.rand(no_of_neurons, self.prev_layer_shape, device = self.device, requires_grad = True)

        ## Add the layer to the model architecture
        self.no_of_layers +=1
        self.layers.append(layer)
        self.prev_layer_shape = no_of_neurons

        ## Add the intiated weights to weights list
        self.weights.append(weights)

        ## Add the activation to the layer
        dict = {'act':layer_activation, 't_norm':t_norm, 's_norm':s_norm}
        self.activations.append(dict)

    @staticmethod
    def compute_layer(self, weight_matrix, input_matrix, layer_number):
        ## Device is GPU
        device = self.device

        if self.activations[layer_number]['act'] == 'AND':
            """
                AND Neuron ==> T(S(x,y))
            """
            if self.activations[layer_number]['s_norm'] == "max":
                z = Activations.s_norm_max("b",device, weight_matrix, input_matrix)
            elif self.activations[layer_number]['s_norm'] == "luka":
                z = Activations.luka_s_norm("b",device, weight_matrix, input_matrix)
            elif self.activations[layer_number]['s_norm'] == "prob":
                z = Activations.prob_s_norm("b",device, weight_matrix, input_matrix)

            if self.activations[layer_number]['t_norm'] == "min":
                z = Activations.t_norm_min("v",device, z = z)
            elif self.activations[layer_number]['t_norm'] == "luka":
                z = Activations.luka_t_norm("v",device, z = z)
            elif self.activations[layer_number]['t_norm'] == "prod":
                z = Activations.prod_t_norm("v",device, z = z)

        elif self.activations[layer_number]['act'] == 'OR':
            print(self.activations[layer_number])
            """
                OR Neuron ==> S(T(x,y))
            """
            if self.activations[layer_number]['t_norm'] == "min":
                z = Activations.t_norm_min("b",device, weight_matrix, input_matrix)
            elif self.activations[layer_number]['t_norm'] == "luka":
                z = Activations.luka_t_norm("b",device, weight_matrix, input_matrix)
            elif self.activations[layer_number]['t_norm'] == "prod":
                z = Activations.prod_t_norm("b",device, weight_matrix, input_matrix)

            if self.activations[layer_number]['s_norm'] == "max":
                z = Activations.s_norm_max("v",device, z = z)
            elif self.activations[layer_number]['s_norm'] == "luka":
                z = Activations.luka_s_norm("v",device, z = z)
            elif self.activations[layer_number]['s_norm'] == "prob":
                z = Activations.prob_s_norm("v",device, z = z)

        self.layers[layer_number] = z


    def train_model(self, inp, output):
        ## Feed Forward

        for i in range(1,self.no_of_layers):
            if i == 0:
                inp = inp
            else:
                inp = self.layers[i-1]

            Model.compute_layer(self,self.weights[i],self.layers[i-1],i)
        pred = self.layers[(self.no_of_layers - 1)]
        loss = pred - output
        loss.backward(torch.empty(loss.size(), device = 'cuda:0'))
        for i in self.weights:
            print(i.grad)

        ## Backpropagation
