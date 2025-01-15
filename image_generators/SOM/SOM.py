import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def highlight_cell(x, y, ax=None, **kwargs):
    """
    Highlights a cell in a grid by drawing a rectangle around it.
    """
    rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    
    def __init__(self, m, n, dim, n_iter, alpha=None, sigma=None, toroidal=False, dist_metric="euclidean"):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
 
        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """
        
        super(SOM, self).__init__()
        # Initialise class attributes
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iter = n_iter
        if alpha == None:
            self.alpha = 0.3
        else:
            self.alpha = alpha
        if sigma == None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = sigma
        self.toroidal = toroidal
        self.dist_metric = dist_metric
        self.t = 0 # Increase by one every forward call.
        self.frames = []
        
        # Initialise the weight matrix TODO:  INITIALISE WEIGHTS BASED ON TRAINING DATA
        self.weights = torch.FloatTensor(m*n, dim).uniform_(0.25, 1)
        arr = np.array([i.ravel() for i in np.meshgrid(np.arange(self.m), np.arange(self.n))])
        self.x, self.y = torch.from_numpy(arr).long()
        #self.x, self.y = torch.LongTensor([i.ravel() for i in np.meshgrid(np.arange(self.m), np.arange(self.n))])
        self.locations = torch.LongTensor(np.column_stack((self.x, self.y)))
        
        self.dist_metric = dist_metric
        if dist_metric == "cosine":
            self.dist = lambda x1, x2: F.cosine_similarity(x1, x2, dim=-1) # Cosine similarity
        else:
            self.dist = nn.PairwiseDistance(p=2) # Euclidean
        
        
        
    def get_weights(self):
        return self.weights
    
    
    def get_location(self):
        return self.locations
    
    
    def get_bmu(self, x):
        """
        Calculates the best matching unit for a given input vector.
        Input: x, a 1-D Tensor of shape [dim].
        Returns: bmu, the index of the best matching unit and bmu_loc, the grid location of the best matching unit.
        """
        
        # Calculate pairwise-distance of data point x with all neurons. P-norm with p=2 is equivalent to euclidean distance.
        dists = self.dist(torch.stack([x for i in range(self.m*self.n)]), self.weights)
        
        # Find neuron with minimum distance from input vector as BMU.
        if self.dist_metric == "cosine":
            _, bmu = torch.max(dists, 0)
            
        else:
            _, bmu = torch.min(dists, 0)
        
        # Get the grid location of BMU.
        bmu_loc = self.locations[bmu,:]
        
        return bmu, bmu_loc



    def update_weights(self, locations, weights, bmu_loc, x, alpha_op, sigma_op):
        """
        Calculates the new weights of the SOM.
        """
        
        dim = len(x)
        length = len(locations)
        
        if self.toroidal:
            # Calculatiung toroidal distance between each nueron and the bmu
            dx = abs(self.x - bmu_loc[0])
            dy = abs(self.y - bmu_loc[1])
            dx[dx > self.m/2] -= self.m
            dy[dy > self.n/2] -= self.n
            distances = torch.stack([dx, dy], 1)
        
        else:
            # Non-toroidal distance calculation
            dx = self.x - bmu_loc[0]
            dy = self.y - bmu_loc[1]
            distances = torch.stack([dx, dy], dim=1).float()
        
        # Calculating the neighbourhood function
        square_distances = torch.pow(distances, 2) # Tensor of square of the distances, shape = (m*n, 2), dtype = int, e.g. [169, 1], ... [36, 324]
        sum_square_distances = torch.sum(square_distances, 1) # Tensor of sum of x and y distances of each neuron to bmu, shape = [m*n], dtype = int, e.g [170, 37, 35, ...]
        scaled_distances = torch.div(sum_square_distances, sigma_op**2) # Divide distances by neighbourhood radius squared, further distances have a higher number, shape = [m*n], dtype = float, e.g. [0.8500, 0.6800, 0.5300, ...]
        negative_scaled_distances = torch.neg(scaled_distances) # Negative of the above neighborhood function, further neurons now have lower number  shape = [m*n], dtype = float, e.g. [-0.8500, -0.6800, -0.5300, ...]
        neighbourhood_scalar = torch.exp(negative_scaled_distances) # Exponential of the scaled distances, since e^negative: will be between 0 and 1, with more negative closer to 0. shape = [m*n], dtype = float, e.g. [0.4300, 0.5100, 0.5900, ...]
        
        # Calculate the new weights
        lr_scalar = alpha_op * neighbourhood_scalar # Learning rate multiplied by the neighbourhood function, shape = [m*n], dtype = float, e.g. [0.1290, 0.1530, 0.1770, ...]
        match_dim_lr_scalars =  torch.stack([lr_scalar[i].repeat(dim) for i in range(length)]) # Necessary for mult; repeats the learning rate scalar for the dimensionality of the weights, shape = [m*n, dim], dtype = float, e.g. [0.1290, 0.1290, 0.1290], ... [0.1530, 0.1530, 0.1530], ...
        data_weight_diff = x - weights # Tensor of the difference between the data point and the weights, shape = [m*n, dim], dtype = float, e.g. [0.1290, 0.1290, 0.1290], ... [0.1530, 0.1530, 0.1530], ...
        delta = torch.mul(match_dim_lr_scalars, data_weight_diff) # Tensor multiplying the difference by the scalar. shape = [m*n, dim], dtype = float, e.g. [0.0852, 0.0518, 0.0238], ... [0.0382, 0.0852, 0.0518], ...
        new_weights = torch.add(weights, delta)
        
        return new_weights



    ### Learning
    def forward(self, x, it, GIF=False):
        
        _, bmu_loc = self.get_bmu(x)
    
        # Calculate the learning rate and neighbourhood radius
        learning_rate = 1 - it/self.n_iter
        alpha_op = self.alpha * learning_rate
        sigma_op = self.sigma * learning_rate
        
        # Update weights
        self.weights = self.update_weights(self.locations, self.weights, bmu_loc, x, alpha_op, sigma_op)
                
        self.t += 1
        

    
    ### Visualisations
    def plot_locations(self, m, n, x, bmu_loc, alpha_op, sigma_op, ax):
        """ 
        Plots locations of neurons. Highlights the BMU, and colours by adjusted weights toward the BMU. 
        Only works if dim of neurons = 3 (RGB) values between 0 and 1.
        """
        
        centroid_grid = np.zeros((m, n, 3))
        centroid_weights = torch.Tensor([0.8]*3*self.m*self.n).reshape(self.m*self.n, 3)
        
        locations = self.get_location()
        centroid_weights = self.update_weights(locations, centroid_weights, bmu_loc, x, alpha_op, sigma_op)
        
        for i, loc in enumerate(locations):
            centroid_grid[loc[0]][loc[1]] = centroid_weights[i]

        
        linewidth=2
        ax.imshow(centroid_grid)
        ax.set_xticks(np.arange(0, n, 1))
        ax.set_yticks(np.arange(0, m, 1))
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, m, 1), minor=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        highlight_cell(bmu_loc[1], bmu_loc[0], ax=ax, color='black', linewidth=linewidth)

        ax.grid(which='minor', color='w', linestyle='-', linewidth=linewidth)
        for i, (x, y) in enumerate(self.locations):
            ax.text(y, x, str(i), color='black', ha='center', va='center', fontsize=m/(m/6))
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.set_title(f"BMU")

    
    
    def plot_som(self, m, n, ax, vectors=[]):
        """ 
        Plots the grid of neuron weights. 
        Only works if dim of neurons = 3 (RGB) values between 0 and 1.
        """
        
        centroid_grid = np.zeros((m, n, 3))
        weights = self.get_weights()
        locations = self.get_location()
        for i, loc in enumerate(locations):
            centroid_grid[loc[0]][loc[1]] = weights[i].numpy()
        
        ax.imshow(centroid_grid)
        ax.set_xticks(np.arange(0, n, 1))
        ax.set_yticks(np.arange(0, m, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"SOM")
        for vector in vectors:
            _, loc = self.get_bmu(vector)
            highlight_cell(loc[1], loc[0], ax=ax, color='black', linewidth=2)


m = 40
n = 40
dim = 3
n_iter = 100



som = SOM(m, n, dim, n_iter, alpha=.8, sigma=12)

som.forward(torch.FloatTensor(3).uniform_(.25, 1), 0)