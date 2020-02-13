improt numpy as np

class Activations:

"""
    Activation Functions used in the module are defined here.
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
