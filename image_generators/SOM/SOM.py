import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    Input for training data should be examples x dimension i.e. a SOM for gene clustering, would be #genes x #samples.
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
        self.weights = torch.FloatTensor(m*n, dim).uniform_(-1, 1)
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
        Calculates the best matching unit for a given single input vector x.
        """
        # shape: [M, D] vs [D] => we replicate x to shape [M, D]
        dists = self.dist(torch.stack([x for _ in range(self.m*self.n)]), self.weights)
        
        if self.dist_metric == "cosine":
            dist, bmu = torch.max(dists, 0)   # similarity => max
        else:
            dist, bmu = torch.min(dists, 0)   # distance => min
        
        if return_dist:
            return dist
        else:
            bmu_loc = self.locations[bmu, :]
            return bmu, bmu_loc

    # ----------------------------------------------------------------
    # Vectorized BMU for entire batch
    # ----------------------------------------------------------------
    def get_bmus_batch(self, vectors):
        """
        Computes BMU indices and BMU locations for all input vectors at once.
        vectors: shape [N, D]
        returns: bmu_idx (shape: [N]), bmu_locs (shape: [N, 2])
        """
        # Expand shapes:
        # weights: [M, D], vectors: [N, D]
        # For Euclidean: compute pairwise distance in a vectorized way
        if self.dist_metric == "euclidean":
            # dist_matrix: shape [N, M]
            # Strategy:  d(X, W)^2 = X^2 + W^2 - 2 XW^T  => then sqrt if needed
            X_sq = (vectors**2).sum(dim=1, keepdim=True)      # [N, 1]
            W_sq = (self.weights**2).sum(dim=1).unsqueeze(0)  # [1, M]
            # XW^T: shape [N, M]
            vectors = vectors.to(self.weights.dtype)
            cross_term = 2 * vectors.matmul(self.weights.t())  # [N, M]
            dist_matrix_sq = X_sq + W_sq - cross_term
            dist_matrix_sq = torch.clamp(dist_matrix_sq, min=1e-9)  # avoid negatives
            dist_matrix = torch.sqrt(dist_matrix_sq)  # [N, M]
            
            # BMU is min distance => argmin
            bmu_idx = dist_matrix.argmin(dim=1)
        
        elif self.dist_metric == "cosine":
            # Using dot-product-based approach for batch, or just use F.cosine_similarity
            # But F.cosine_similarity expects [N, D], [N, D], so we do some broadcasting.
            # Alternatively, do a manual approach:
            #   cos_sim = (X · W) / (||X|| ||W||)
            # for each pair in a batched manner
            X_norm = F.normalize(vectors, dim=1)         # [N, D]
            W_norm = F.normalize(self.weights, dim=1)    # [M, D]
            # shape [N, M] => X_norm @ W_norm.T
            sim_matrix = X_norm.matmul(W_norm.t())
            # BMU is max similarity => argmax
            bmu_idx = sim_matrix.argmax(dim=1)
        
        # Now map bmu_idx => bmu_locs
        bmu_locs = self.locations[bmu_idx]  # shape: [N, 2]
        return bmu_idx, bmu_locs

    def get_bmus(self, vectors):
        """
        Just return the BMU locations as a NumPy array [N, 2].
        """
        _, bmu_locs = self.get_bmus_batch(vectors)
        return bmu_locs.cpu().numpy()  # ensure it's a NumPy array
    
    def get_avg_bmu_dist(self, vectors):
        """
        Compute average distance of each vector to its BMU in a vectorized way.
        """
        if self.dist_metric == "euclidean":
            # Let’s reuse the partial steps from get_bmus_batch to compute distances
            X_sq = (vectors**2).sum(dim=1, keepdim=True)      # [N, 1]
            W_sq = (self.weights**2).sum(dim=1).unsqueeze(0)  # [1, M]
            vectors = vectors.to(self.weights.dtype)
            cross_term = 2 * vectors.matmul(self.weights.t())  # [N, M]
            dist_matrix_sq = X_sq + W_sq - cross_term
            dist_matrix_sq = torch.clamp(dist_matrix_sq, min=1e-9)
            dist_matrix = torch.sqrt(dist_matrix_sq)  # [N, M]
            
            # For each row, get the min distance to a neuron
            min_dist, _ = dist_matrix.min(dim=1)  # shape [N]
            return min_dist.mean()                # scalar
        else:
            # Cosine => we want 1 - similarity if we treat "distance" as such
            X_norm = F.normalize(vectors, dim=1)
            W_norm = F.normalize(self.weights, dim=1)
            sim_matrix = X_norm.matmul(W_norm.t())  # shape [N, M]
            max_sim, _ = sim_matrix.max(dim=1)
            # "Distance" can be (1 - similarity) or something else
            # If you truly want an average distance metric, define it consistently:
            avg_dist = (1.0 - max_sim).mean()
            return avg_dist

    def get_bmu_changes(self, vectors, prev_bmu_list):
        """
        Compare the current BMU locations with the previous BMU list
        in a vectorized way.
        """
        current_bmus = self.get_bmus(vectors)  # shape [N, 2], np.array
        changes = np.sum(~np.all(current_bmus == prev_bmu_list, axis=1))
        return current_bmus, changes
    
    
    
    def update_weights(self, weights, bmu_loc, x, alpha_op, sigma_op):
        """
        Calculates the new weights of the SOM with optimized operations.
        """
        # Calculate distances based on toroidal or non-toroidal grid
        dx = self.x - bmu_loc[0]
        dy = self.y - bmu_loc[1]

        if self.toroidal:
            dx = torch.where(dx > self.m / 2, self.m - dx, dx)
            dy = torch.where(dy > self.n / 2, self.n - dy, dy)

        # Compute squared Euclidean distance
        distances = dx**2 + dy**2  # Shape: [m*n], squaring removes the sign here.

        # Neighborhood function (Gaussian)
        neighbourhood_scalar = torch.exp(-distances / (2 * sigma_op**2))  # Shape: [m*n]

        # Scale the learning rate by the neighborhood function
        lr_scalar = alpha_op * neighbourhood_scalar  # Shape: [m*n]

        # Compute weight updates in a single step
        delta = lr_scalar[:, None] * (x - weights)  # Broadcasting scalar to match dimensions

        # Update weights
        new_weights = weights + delta

        return new_weights



    ### Learning
    def forward(self, x, step):
        
        _, bmu_loc = self.get_bmu(x)

    
        alpha_op = self.alpha * (self.decay**step)
        sigma_op = self.sigma * (self.decay**step)

        # Update weights
        self.weights = self.update_weights(self.locations, self.weights, bmu_loc, x, alpha_op, sigma_op)
    