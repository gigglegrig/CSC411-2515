import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)


class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

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
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.update = 0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.update = self.beta * self.update - self.lr * grad
        params += self.update
        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        loss = 1 - X.dot((self.w).T) * y
        loss[loss < 0] = 0

        return loss

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        N = X.shape[0]
        w = [x for x in self.w]
        w[-1] = 0
        loss = self.hinge_loss(X, y)
        loss_grad = self.w - self.c / N * np.sum(X[loss > 0] * y[loss > 0][:, np.newaxis], axis=0)

        return loss_grad

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        y = np.sign(X.dot(self.w))
        return y


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''

    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for i in range(steps):
        # Optimize and update the history
        w = optimizer.update_params(w, func_grad(w))
        w_history.append(w)
    return w_history


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    M = train_data.shape[1]
    params = np.zeros(M)
    svm_clf = SVM(penalty, M)
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)

    for i in range(iters):
        # get batch X,y
        X, y = batch_sampler.get_batch()
        params = optimizer.update_params(params, svm_clf.grad(X, y))
        svm_clf.w = params

    return svm_clf


def plot_w(w):
    w_matrix = np.reshape(w, (28, 28))
    plt.imshow(w_matrix, cmap='gray')
    plt.show()


if __name__ == '__main__':
    # Part 1
    # a=1 b=0
    optimizer1 = GDOptimizer(1, 0)
    w1 = optimize_test_function(optimizer1)
    # a=1 b=0.9
    optimizer2 = GDOptimizer(1, 0.9)
    w2 = optimize_test_function(optimizer2)

    x_ticks = list(range(201))
    plt.plot(x_ticks, w1, label='b=0.9')
    plt.plot(x_ticks, w2, label='b=0')
    plt.title('With/Without Momentum')
    plt.legend()
    plt.show()
    # Part 2 & 3
    # Load data
    train_data, train_targets, test_data, test_targets = load_data()
    train_size, test_size = train_data.shape[0], test_data.shape[0]
    # add bias column as last column
    train_biased = np.append(train_data, np.ones(train_size)[:, np.newaxis], axis=1)
    test_biased = np.append(test_data, np.ones(test_size)[:, np.newaxis], axis=1)

    # set optimizers for 2 models
    svm_optimizer1 = GDOptimizer(0.01, 0)
    svm_optimizer2 = GDOptimizer(0.01, 0.1)
    # return the trained models with each optimizer
    svm_clf1 = optimize_svm(train_biased, train_targets, penalty=1, optimizer=svm_optimizer1, batchsize=100, iters=500)
    # return loss1
    train_loss1 = svm_clf1.hinge_loss(train_biased, train_targets)
    test_loss1 = svm_clf1.hinge_loss(test_biased, test_targets)
    # predict and return train accuracy
    train_pred1 = svm_clf1.classify(train_biased)
    train_accuracy1 = np.mean(train_pred1 == train_targets)
    # predict and return test accuracy
    test_pred1 = svm_clf1.classify(test_biased)
    test_accuracy1 = np.mean(test_pred1 == test_targets)
    # plot weights as graph
    plot_w(svm_clf1.w[:-1])
    plt.title('w with beta=0')
    print('For Model with beta=0:\n \
          Train accuracy: {}\n \
          Test accuracy: {}\n \
          Avg train hinge loss: {}\n\
          Avg test hinge loss: {}\n'.format(train_accuracy1, test_accuracy1, \
                                            np.mean(train_loss1), np.mean(test_loss1)))
    # train model with optimizer2
    svm_clf2 = optimize_svm(train_biased, train_targets, penalty=1, optimizer=svm_optimizer2, batchsize=100, iters=500)
    # return the train and test hinge losses
    train_loss2 = svm_clf2.hinge_loss(train_biased, train_targets)
    test_loss2 = svm_clf2.hinge_loss(test_biased, test_targets)
    # predict and return train accuracy
    train_pred2 = svm_clf1.classify(train_biased)
    train_accuracy2 = np.mean(train_pred2 == train_targets)
    # predict and return test accuracy
    test_pred2 = svm_clf2.classify(test_biased)
    test_accuracy2 = np.mean(test_pred2 == test_targets)
    # plot the weights as graph
    plot_w(svm_clf2.w[:-1])
    plt.title('w with beta=0.1')
    print('For Model with beta=0.1:\n\
          Train accuracy: {}\n\
          Test accuracy: {}\n\
          Avg train hinge loss: {}\n\
          Avg test hinge loss: {}\n'.format(train_accuracy2, test_accuracy2, \
                                            np.mean(train_loss2), np.mean(test_loss2)))