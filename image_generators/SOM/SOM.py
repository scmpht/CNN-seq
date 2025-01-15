import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    
    def __init__(self, m, n, dim, decay, alpha=None, sigma=None, toroidal=False, dist_metric="euclidean"):
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
        self.decay = decay
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
        self.step = 0 
        
        # Initialise the weight matrix TODO:  INITIALISE WEIGHTS BASED ON TRAINING DATA
        self.weights = torch.FloatTensor(m*n, dim).uniform_(0, 1)
        arr = np.array([i.ravel() for i in np.meshgrid(np.arange(self.m), np.arange(self.n))])
        self.x, self.y = torch.from_numpy(arr).long()
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
    
    
    def get_bmu(self, x, return_dist=False):
        """
        Calculates the best matching unit for a given input vector.
        Input: x, a 1-D Tensor of shape [dim].
        Returns: bmu, the index of the best matching unit and bmu_loc, the grid location of the best matching unit.
        """
        
        # Calculate pairwise-distance of data point x with all neurons. P-norm with p=2 is equivalent to euclidean distance.
        dists = self.dist(torch.stack([x for i in range(self.m*self.n)]), self.weights)
        
        # Find neuron with minimum distance from input vector as BMU.
        if self.dist_metric == "cosine":
            dist, bmu = torch.max(dists, 0)
            
        else:
            dist, bmu = torch.min(dists, 0)
        
        # Get the grid location of BMU.
        bmu_loc = self.locations[bmu,:]
        
        # For calculating average distance metrics
        if return_dist:
            return dist
        else:
            return bmu, bmu_loc
    
    
    
    def get_avg_bmu_dist(self, vectors):
        cum_dist = 0
        for vector in vectors:
            dist = self.get_bmu(vector, return_dist=True)
            cum_dist += dist
        return cum_dist/vectors.shape[0]
    
    
    def get_bmus(self, vectors):
        """
        Computes the BMU locations for all input vectors and returns them as a 2D NumPy array.
        """
        bmus = np.zeros((len(vectors), 2)) 
        for idx, vector in enumerate(vectors):
            _, loc = self.get_bmu(vector)
            bmus[idx] = loc.numpy()
        return bmus
            
    
    def get_bmu_changes(self, vectors, prev_bmu_list):
        """
        Compares the current BMU locations with the previous BMU list.
        Returns the updated BMU list and the count of changes.
        """
        current_bmus = self.get_bmus(vectors)
        # Compare current BMUs with previous BMUs
        changes = np.sum(~np.all(current_bmus == prev_bmu_list, axis=1))
        return current_bmus, changes
        



    def update_weights(self, locations, weights, bmu_loc, x, alpha_op, sigma_op):
        """
        Calculates the new weights of the SOM.
        """
        
        dim = len(x)
        length = len(locations)
        
        if self.toroidal:
            # Calculatiung toroidal distance between each neuron and the bmu
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
    def forward(self, x):
        
        bmu, bmu_loc = self.get_bmu(x)
    
        alpha_op = self.alpha * (self.decay**self.step)
        sigma_op = self.sigma * (self.decay**self.step)

        # Update weights
        self.weights = self.update_weights(self.locations, self.weights, bmu_loc, x, alpha_op, sigma_op)
        
        self.step += 1
    