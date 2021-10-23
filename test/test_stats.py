import pytest
import numpy as np
from src.stats import calculate_mean, calculate_cov

@pytest.fixture
def sample_data():
    x = np.random.rand(5,3) # You can change this sample data should you wish to
    yield x

@pytest.fixture
def mean_of_data(sample_data):
    yield np.mean(sample_data, axis=0)

def test_mean(sample_data):
    """
    GIVEN a sample data set
    WHEN the mean
    :param sample_data:
    :return:
    """
    mean = calculate_mean(sample_data)
    assert np.allclose(mean, np.mean(sample_data, axis=0))


def test_cov(sample_data, mean_of_data):
    cov_mat = calculate_cov(sample_data, mean_of_data)
    cov = np.cov(sample_data, rowvar=False)
    assert np.allclose(cov_mat, cov)
