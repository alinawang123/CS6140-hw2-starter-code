import numpy as np
from .stats import calculate_mean, calculate_cov


class PCA:
    def __init__(self, n_components=None, mean=None, cov=None):
        self.mean = None
        self.cov = None
        self.n_components = n_components

    def calculate_pca(self, x_data):
        """
        Given x_data as the input data calculate
        a matrix to calculate a PCA matrix
        :return: W
        """
        self.mean = calculate_mean(x_data)
        x_data_centered = x_data - self.mean
        self.cov = calculate_cov(x_data_centered, self.mean)
        
        # Compute eigenvectors & eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)

        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)


        # Sort eigenvectors by descending eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Choose the top n_components eigenvectors to form W
        if self.n_components is None:
            self.n_components = x_data.shape[1]
        W = eigenvectors[:, :self.n_components]
        
        return W