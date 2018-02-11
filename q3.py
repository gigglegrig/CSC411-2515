import numpy as np
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt

BATCHES = 50


class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]  # n= 506
        self.features = data.shape[1]  # d = 13
        self.batch_size = batch_size
        self.X = data
        self.y = targets
        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.X, indices, 0)
        y_batch = self.y[indices]
        return X_batch, y_batch


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)


# TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    sample = X.shape[0]
    gradient = (2 * X.T.dot(X).dot(w) - 2 * X.T.dot(y)) / sample  # d*1
    return gradient
    # raise NotImplementedError()


# Customized helper function to find mini_batch gradient
def mini_batch(X, y, w, K, m):
    '''
    Input: X, y, w, K, m
    Output: True gradient and mini-batch gradient of m sized batch for K times.
    '''
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, m)
    # calculatre mini-batch grad of batch size m for K times, take the avg.
    batch_grad = np.zeros(w.shape)  # initiallize mini-batch grad
    for i in range(K):
        X_b, y_b = batch_sampler.get_batch()  # get a batch of size 50 randomly
        batch_grad += lin_reg_gradient(X_b, y_b, w)
    batch_grad = batch_grad / K
    return batch_grad


# Customized helper function to find variance for 1st item in w'.
def elemental_var(X, y, w, K, m):
    '''
    Input entire sample space X, y; Iteration # K; mini-batch size m
    Output variance of the index=1 element in K iterations
    '''
    batch_sampler = BatchSampler(X, y, m)
    w1_grad = np.zeros(K)
    for i in range(K):
        X_b, y_b = batch_sampler.get_batch()
        w1_grad[i] = lin_reg_gradient(X_b, y_b, w)[1]  # get w'[1]
    var = sum([(x - np.mean(w1_grad)) ** 2 for x in w1_grad]) / K
    return var


def main():
    # Load data and randomly initialise weights
    np.random.seed(1)
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data

    # True gradient
    true_grad = lin_reg_gradient(X, y, w)
    print('True_grad :\n {}'.format(true_grad))

    # Part 5
    # Find mini_batch gradient with K and m
    batch_grad = mini_batch(X, y, w, K=500, m=50)
    print('Mini-batch grad:\n {}'.format(batch_grad))

    euclidean = np.sum((batch_grad - true_grad) ** 2)
    print('squared distance metric: {}'.format(euclidean))
    cos_theta = cosine_similarity(true_grad, batch_grad)
    print('cosine similarity: {}'.format(cos_theta))

    # Part 6
    '''
    -build m list [1,400]
    for 1-400
        compute variance for 1st element of w'
    -take log variance and logm and plot
    '''
    m = list(range(1, 401))
    variance_list = np.zeros(len(m))
    for i in range(400):
        variance_list[i] = elemental_var(X, y, w, K=500, m=m[i])
    log_var = np.log(variance_list)
    log_m = np.log(m)
    plt.plot(log_m, log_var)
    plt.xlabel('log(m)')
    plt.ylabel('log(variance)')
    plt.title('log(var) vs. log(m)')
    plt.show()


if __name__ == '__main__':
    main()