from stats import calculate_mean, calculate_cov


class PCA:
    def __init__(self, mean=None, cov=None):
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
        self.cov = calculate_cov(x_data, self.mean)
        
        # Compute eigenvectors & eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)
        
        # Sort eigenvectors by descending eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Choose the top n_components eigenvectors to form W
        if self.n_components is None:
            self.n_components = x_data.shape[1]
        W = eigenvectors[:, :self.n_components]
        
        return W