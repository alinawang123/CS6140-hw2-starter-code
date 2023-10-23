from stats import calculate_mean, calculate_cov

class GaussianModel:
    def __init__(self, mean=None, cov=None):
        self.mean = mean
        self.cov = cov
        if mean is not None:
            self.d = len(mean)  # Set this to the feature dimension
        else:
            self.d = None

    def calculate_log_likelihood(self, x):
        """
        Calculate the log-likelihood for
        :param x: A data point
        :return: log-likelihood
        """
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        term1 = -0.5 * self.d * np.log(2 * np.pi)
        term2 = -0.5 * np.log(det_cov)
        diff = x - self.mean
        term3 = -0.5 * diff.T @ inv_cov @ diff
        return term1 + term2 + term3


