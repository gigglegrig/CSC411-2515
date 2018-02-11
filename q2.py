# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses per fold, one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j, tau in enumerate(taus):
        predictions = np.array([LRLS(x_test[i, :].reshape(d, 1), x_train, y_train, tau) \
                                for i in range(N_test)])
        losses[j] = ((predictions.flatten() - y_test.flatten()) ** 2).mean()
    return losses


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter, single scalar per call.
           lam is the regularization parameter, CONST
    output is y_hat the prediction on test_datum, scalar valued.
    '''
    Ai = (np.sum((x_train - test_datum.T) ** 2, axis=1) / (-2 * (tau ** 2)))  # Ai.shape = (N*1)
    B = np.amax(Ai)
    ai = np.exp(Ai - B) / np.exp(logsumexp(Ai - B))
    A = np.diagflat(ai)
    LHS = x_train.T.dot(A).dot(x_train) + lam * np.eye(x_train.shape[1])
    RHS = x_train.T.dot(A).dot(y_train)
    w = np.linalg.solve(LHS, RHS)  # shape: (14*1)
    y_hat = w.dot(test_datum)

    return y_hat


def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    losses = np.zeros(taus.shape[0])
    fold_size = round(N / k)  # 506/5=101
    # In each fold, choose indice from the randomlized idx with random.seed(0)
    for i in range(k):  # 0 1 2 3 4
        test_idx = idx[i * fold_size: (i + 1) * fold_size]
        train_idx = [x for x in idx if x not in test_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
        train_x = x[train_idx]
        train_y = y[train_idx]
        losses += run_on_fold(test_x, test_y, train_x, train_y, taus)  # += for avg losses of each fold

    return losses / k


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(taus, losses)
    plt.xlabel("taus")
    plt.ylabel("losses")
    plt.title("avg losses vs. taus")
    plt.show()
    print("min loss = {}".format(losses.min()))