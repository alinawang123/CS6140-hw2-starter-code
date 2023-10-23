import pytest
import numpy as np
from src.gaussian import GaussianModel
from src.stats import calculate_mean, calculate_cov

def test_gaussian_log_likelihood():
    mean = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 1.2]])
    model = GaussianModel(mean, cov)
    x = np.array([0.5, -0.3])
    log_likelihood = model.calculate_log_likelihood(x)
    assert np.isclose(log_likelihood, -2.8033088488)


def test_gaussian_invalid_input():
    mean = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 1.2]])
    x = np.array([0.5])  # this input is of invalid shape
    
    model = GaussianModel(mean, cov)

    # Assert that an exception is raised with invalid input shape
    with pytest.raises(ValueError, match="Input data point dimension does not match the model's dimension."):
        log_likelihood = model.calculate_log_likelihood(x)