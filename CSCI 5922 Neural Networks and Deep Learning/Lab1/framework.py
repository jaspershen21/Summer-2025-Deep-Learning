import torch
import numpy as np
import tqdm

from activations import *
from mnist import *

class MLP:
    # '''
    # This class should implement a generic MLP learning framework. The core structure of the program has been provided for you.
    # But, you need to complete the following functions:
    # 1: initialize()
    # 2: forward(), including activations
    # 3: backward(), including activations
    # 4: TrainMLP()
    # '''
    def __init__(self, layer_sizes: list[int]):
        #Storage for model parameters
        self.layer_sizes: list[int] = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights: list[torch.tensor] = []
        self.biases: list[torch.tensor] = []

        #Temporary data
        self.features: list[torch.tensor] = []

        #hyper-parameters w/ default values
        self.learning_rate: float = 1
        self.batch_size: int = 1
        self.activation_function: callable[[torch.tensor], torch.tensor] = ReLU

    def set_hp(self, lr: float, bs: int, activation: object) -> None:
        self.learning_rate = lr
        self.batch_size = bs
        self.activation_function = activation

        return

    def initialize(self) -> None:
        #Complete this function
        
        # '''
        # initialize all biases to zero, and all weights with random sampling from a uniform distribution.
        # This uniform distribution should have range +/- sqrt(6 / (d_in + d_out))
        # '''
        for i in range(self.num_layers):
            # determine current and prior layer sizes
            dim_current = self.layer_sizes[i+1]
            dim_prior = self.layer_sizes[i]

            # initialize biases as zeros
            self.biases.append(torch.zeros(1,dim_current))

            # compute Xavier Uniform Initialization by hand for weights
            rand_lim = np.sqrt(6/(dim_current + dim_prior))
            self.weights.append((torch.rand(dim_prior, dim_current)*2 - 1)*rand_lim)  # might error due to improper multiplication?



        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        #Complete this function 
        '''
        This function should loop over all layers, forward propagating the input via:
        x_i+1 = f(x_iW + b)
        Remember to STORE THE INTERMEDIATE FEATURES!
        '''

        # store input as first 'features'
        self.features.append(x)

        for i in range(self.num_layers):
            # pass x through weights, biases, and activation function
            x = self.activation_function.forward(torch.matmul(x, self.weights[i]) + self.biases[i])

            # append x to features
            self.features.append(x)
        
        # compute softmax
        # x = torch.exp(x) / torch.sum(torch.exp(x))

        return x
    

    def backward(self, delta: torch.tensor) -> None:
        #Complete this function
        '''
        This function should backpropagate the provided delta through the entire MLP, and update the weights according to the hyper-parameters
        stored in the class variables.
        '''

        batch_size = delta.shape[0]  # get batch size

        dCE_dO = [None] * self.num_layers
        dCE_dW = [None] * self.num_layers
        dCE_db = [None] * self.num_layers

        # set final error to delta
        dCE_dO[-1] = delta # 512 x 10

        # backpropagate error
        for i in reversed(range(self.num_layers)):
            # compute and multiply gradient of o wrt node value (z)
            dCE_dZ = self.activation_function.backward(dCE_dO[i], self.features[i+1]) # 512 x 10
            
            # combine dCE_dZ and dZ_dW to get dCE_dW using batch matrix multiplication
            # gradient of node value wrt preceding weights (dZ_dW) is output of previous nodes (out*weight + bias = z)
            # dCE_dW[i] = torch.matmul(dCE_dZ.T, self.features[i]).T / batch_size # 256 x 10

            # other options 
            dCE_dW[i] = torch.bmm(dCE_dZ.unsqueeze(2), self.features[i].unsqueeze(1))  # 512 x 10 x 256
            dCE_dW[i] = torch.transpose(dCE_dW[i], dim0=1, dim1=2) # 512 x 256 x 10
            # dCE_dW[i] = torch.matmul(self.features[i].T, dCE_dZ) / batch_size # 256 x 10

            # compute bias gradients (same as node / feature gradient)
            dCE_db[i] = dCE_dZ.mean(dim=0) # 1 x 10

            if i > 0:
                dCE_dO[i - 1] = torch.matmul(dCE_dZ, self.weights[i].T)  # 512 x 256

        # update weights & biases
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * dCE_dW[i]#.mean(dim=0)
            self.biases[i] -= self.learning_rate * dCE_db[i]
        
        return


def TrainMLP(model: MLP, x_train: torch.tensor, y_train: torch.tensor) -> MLP:
    #Complete this function
    '''
    This function should train the MLP for 1 epoch, using the provided data and forward/backward propagating as necessary.
    '''

    #set up a random sampling of the data
    bs = model.batch_size
    N = x_train.shape[0]
    rng = np.random.default_rng()
    idx = rng.permutation(N)

    #variable to accumulate total loss over the epoch
    L = 0
    A = 0

    for i in tqdm.tqdm(range(N // bs)):
        x = x_train[idx[i * bs:(i + 1) * bs], ...]
        y = y_train[idx[i * bs:(i + 1) * bs], ...]

        #forward propagate and compute loss (l) here
        y_hat = model.forward(x=x)
        p = torch.exp(y_hat)
        p /= torch.sum(p, dim = 1, keepdim = True)
        l = -1 * torch.sum(y * torch.log(p))
        L += l
        
        #backpropagate here
        model.backward(p-y)

        A += torch.sum(torch.where(torch.argmax(p, dim = 1) == torch.argmax(y, dim = 1), 1, 0))

    print("Train Loss:", L / ((N // bs) * bs), "Train Accuracy: {:.2f}%".format(100 * A / ((N // bs) * bs)))


def TestMLP(model: MLP, x_test: torch.tensor, y_test: torch.tensor) -> tuple[float, float]:
    bs = model.batch_size
    N = x_test.shape[0]

    rng = np.random.default_rng()
    idx = rng.permutation(N)

    L = 0
    A = 0

    for i in tqdm.tqdm(range(N // bs)):
        x = x_test[idx[i * bs:(i + 1) * bs], ...]
        y = y_test[idx[i * bs:(i + 1) * bs], ...]

        y_hat = model.forward(x)
        p = torch.exp(y_hat)
        p /= torch.sum(p, dim = 1, keepdim = True)
        l = -1 * torch.sum(y * torch.log(p))
        L += l
        
        A += torch.sum(torch.where(torch.argmax(p, dim = 1) == torch.argmax(y, dim = 1), 1, 0))

    print("Test Loss:", L / ((N // bs) * bs), "Test Accuracy: {:.2f}%".format(100 * A / ((N // bs) * bs)))

def normalize_mnist() -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    '''
    This function loads the MNIST dataset, then normalizes the "X" values to have zero mean, unit variance. 
    '''

    #IMPORTANT!!!#
    #UPDATE THE PATH BELOW!#
    base_path = '/home/kieran/Documents/Spring 2025/DeepLearning/Lab1/MNIST/'
    #^^^^^^^^#
    #Update this to point to your downloaded MNIST files.
    #E.g., if you've downloaded the data to {SomePath}/Downloads, you would put {SomePath}/Downloads/MNIST/


    mnist = MnistDataloader(base_path + "train-images.idx3-ubyte", base_path + "train-labels.idx1-ubyte",
                            base_path + "t10k-images.idx3-ubyte", base_path + "t10k-labels.idx1-ubyte")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_mean = torch.mean(x_train, dim = 0, keepdim = True)
    x_std = torch.std(x_train, dim = 0, keepdim = True)

    x_train -= x_mean
    x_train /= x_std
    x_train[x_train != x_train] = 0

    x_test -= x_mean
    x_test /= x_std
    x_test[x_test != x_test] = 0

    return x_train, y_train, x_test, y_test

def main():
    '''
    This is an example of how to use the framework when completed. You can build off of this code to design your experiments for part 2.
    '''

    x_train, y_train, x_test, y_test = normalize_mnist()

    '''
    For the experiment, adjust the list [784,...,10] as desired to test other architectures.
    You are encouraged to play around with any of the following values if you so desire:
    E, lr, bs, activation
    '''

    model = MLP([784, 256, 10])
    model.initialize()
    model.set_hp(lr = 1e-6, bs = 512, activation = ReLU)

    E = 25
    for _ in range(E):
        TrainMLP(model, x_train, y_train)
        TestMLP(model, x_test, y_test)

if __name__ == "__main__":
    main()