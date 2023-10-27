import pytest
import numpy as np
from src.stats import calculate_mean, calculate_cov

@pytest.fixture(params = [(5,3), (3,3), (10,5), (1,5), (5, 1)])
def sample_data(request):
    """
    Generate a sample data matrix with n_rows = number of examples,
    n_cols = number of features
    :yields: the data matrix
    """
    shape = request.param
    x = np.random.rand(*shape) # You can change this sample data should you wish to
    yield x

@pytest.fixture
def mean_of_data(sample_data):
    yield np.mean(sample_data, axis=0)

def test_mean(sample_data):
    """
    GIVEN a sample data set
    WHEN the mean is calculated using calculate_mean
    THEN the calculated mean is close to np.mean
    :param sample_data:
    """
    mean = calculate_mean(sample_data)
    assert np.allclose(mean, np.mean(sample_data, axis=0))


def test_cov(sample_data, mean_of_data):
    """
    GIVEN a sample data set
    WHEN the covariance matrix is calculated using calculate_cov
    THEN the calculated covariance matrix is close to that given by np.cov
    :param sample_data:
    """
    cov_mat = calculate_cov(sample_data, mean_of_data)

    if sample_data.shape[0] == 1:
        # If there's only one data point, the covariance matrix should be all zeros
        cov = np.zeros((sample_data.shape[1], sample_data.shape[1]))
    else:
        cov = np.cov(sample_data, rowvar=False)
    assert np.allclose(cov_mat, cov)
