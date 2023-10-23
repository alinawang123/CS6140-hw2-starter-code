import numpy as np


def calculate_mean(x_data):
    """
    This function uses numpy to calculate the mean vector of a dataset
    :param x_data: a 2-D numpy array with n_examples as rows and n_features as columns
    :return: mean_vec
    Note, you may not use np.mean to calculate the mean vector
    """
    sum_vec = np.sum(x_data, axis = 0)
    mean_vec = sum_vec/x_data.shape[0]
    
    return mean_vec


def calculate_cov(x_data, mean_vec):
    """
    This function uses numpy to calculate the covariance matrix of a dataset
    :param x_data: a 2-D numpy array with n_examples as rows and n_features as columns
    :return: cov_mat
    Note, you may not use np.cov to calculate the covariance matrix
    """
    if x_data.shape[0] == 1:
        return np.zeros((x_data.shape[1], x_data.shape[1]))
    deviation_matrix = x_data - mean_vec
    cov_mat = deviation_matrix.T.dot(deviation_matrix)/(x_data.shape[0]-1)
   
    return cov_mat

