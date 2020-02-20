import torch
import numpy as np
from skfuzzy.membership import trimf




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
        z = 1/(1 + np.exp(-z))
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
            a = torch.randn(weight_matrix.size()[0], weight_matrix.size()[1], device = device)
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
            a = torch.randn(weight_matrix.size()[0], weight_matrix.size()[1], device = device)
            j = 0
            for i in weight_matrix:
                a[j] = torch.mul(i, input_matrix.squeeze(1))
                j+=1
            return a
        elif mode == "v":
            # print("Input for Z",z)
            z = torch.prod(z, axis = 1)
            z = z.view(-1,1)
            return z

    # Luckasiewickz T Norm
    @staticmethod
    def luka_t_norm(mode, device, weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            a = torch.rand(weight_matrix.size()[0], weight_matrix.size()[1], device =device)
            j=0
            for i in weight_matrix:
                a[j] = torch.max(i + input_matrix.squeeze(1) - torch.ones(weight_matrix.size()[1], device = device),torch.zeros(weight_matrix.size()[1], device = device))
                j+=1
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
            a = torch.randn(weight_matrix.size()[0],weight_matrix.size()[1],device = device)
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
        # print("Weights", weight_matrix)
        # print("Input Mattrix", input_matrix)
        if mode == "b":
            a = torch.zeros(weight_matrix.size()[0], weight_matrix.size()[1], device = device)
            j=0
            for i in weight_matrix:
                a[j] = torch.sub((i + input_matrix.squeeze(1)),(i*input_matrix.squeeze(1)))
                # print(a[j])
                j+=1
            # print(a)
            return a
        elif mode == "v":
            z = torch.sum(z, axis = 1) - torch.prod(z, axis = 1)
            z = z.view(-1,1)
            return z

    # Luckasiewickz S Norm
    def luka_s_norm(mode,device,weight_matrix = 0, input_matrix = 0, z = 0):
        if mode == "b":
            a = torch.rand(weight_matrix.size()[0], weight_matrix.size()[1], device =device)
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

        ## Gradients of Layers
        self.grad = []

        ## Garbage Variables
        self.prev_layer_shape = 0
        self.iteration = 0
        self.iterationt = 0





    def add_layer(self, no_of_neurons, layer_activation, t_norm = "prod", s_norm = "prob"):
        """
        no_of_neurons ==> Represent the Number of Neurons in that particular layer
        activation ==> Represent the activation to be used for that particular layer
        """
        ## Create the current layer
        # layer = torch.empty(no_of_neurons, 1,device = self.device, requires_grad = True)

        if self.no_of_layers == 0:
            self.prev_layer_shape = self.input_shape

        ## Initialize weights between the current layer and previous layer
        weights = (torch.rand(no_of_neurons, self.prev_layer_shape, device = self.device, requires_grad = True))


        ## Add the layer to the model architecture
        self.no_of_layers +=1
        # self.layers.append(layer)
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

        self.layers.append(z)



    def train_model(self, X, Y, lr = 0.00001):
        print("No of Layers: ", self.no_of_layers)
        self.intermediate = []

        for epoch in range(100):
            overall_loss = 0
            print("Epoch:",epoch+1)
            for i in range(len(X)):
                self.layers = []
                inp = X.iloc[i,:]
                output = Y.iloc[i]
                ## PreProcessing for one input
                a = torch.tensor(inp)
                # a = torch.from_numpy(trimf(a,[0,3,10]))
                a = a.to(device = torch.device('cuda'))
                a = a.view(-1,1)
                output = torch.tensor(output, device = self.device)
                # print("Input Number:",i)
                ## Feed Forward
                for layer in range(self.no_of_layers):
                    if layer == 0:
                        inpt = a
                    else:
                        inpt = self.layers[layer-1]

                    Model.compute_layer(self,self.weights[layer],inpt,layer)


                pred = self.layers[-1]

                ## Loss Function
                loss = (pred - output)**2
                overall_loss +=torch.sum(loss) ## Incase if the output is multidimensional vector
                overall_loss = overall_loss/len(X)

            ## Back Propgation
            overall_loss.backward(retain_graph = True)
            j = 0

            for w in self.weights:
                    w.data -= (lr*((w.grad-torch.min(w.grad))/(torch.max(w.grad)-torch.min(w.grad))))
                    w.grad.zero_()
                    w.data = (w.data - torch.min(w.data))/(torch.max(w.data) - torch.min(w.data))


            print("MSE: ", overall_loss.item())
