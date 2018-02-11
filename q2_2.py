'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for label in set(train_labels):
        means[int(label)] = np.mean(train_data[train_labels == label], axis=0)

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    diag_add = np.identity(64)*0.01
    means = compute_mean_mles(train_data, train_labels)
    for label in set(train_labels):
        a = train_data[train_labels==label] - means[int(label)]
        n = train_data[train_labels==label].shape[0]
        covariances[int(label)] = a.T.dot(a)/n + diag_add
        assert covariances[int(label)].shape == (64,64)
    # Compute covariances
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    diags = []
    for i in range(10):
        cov_diag = np.log(np.diag(covariances[i])) #64*64 log of diagonal elements
        diags.append(np.reshape((cov_diag),(8,8)))
    concat = np.concatenate(diags, 1)
    plt.imshow(concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    digits = np.reshape(digits,(-1,64))
    N = digits.shape[0]
    D = 64
    gen_likelihood = np.zeros((N,10))
    for i,digit in enumerate(digits):
        temp = np.zeros(10)
        for j in range(10): # retuen a p(x|y=k) for each of k in [0,9]
            inv_cov = np.linalg.solve(covariances[j], np.eye(64))
            temp[j] = (-D/2)*np.log(2*np.pi)- (1/2)*np.log(np.linalg.det(covariances[j])) \
                        - (1/2)*(digit-means[j]).T.dot(inv_cov).dot(digit-means[j])
#             print(temp[j])
        gen_likelihood[i] = temp

    return gen_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
#     digits = np.reshape(digits, (-1,64))
#     N = digits.shape[0]
#     con_likelihood = np.zeros((N,10))
#     for i,digit in enumerate(digits):
#         gen_likelihood_arr = generative_likelihood(digit, means, covariances)
#         log_total_prob = np.log(np.sum(np.exp(gen_likelihood_arr)))
#         con_likelihood[i] = gen_likelihood_arr - log_total_prob

    gen_likelihood = generative_likelihood(digits, means, covariances)
    log_total_prob = np.log(np.sum(np.exp(gen_likelihood), axis=1))
    con_likelihood = gen_likelihood - log_total_prob[:, np.newaxis]
    return con_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )
        1*10

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    N = digits.shape[0]
    total = 0
    for n in range(N):
        total = total + cond_likelihood[n][int(labels[n])]
    average = total/N
    return average

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    max_idx = np.argmax(cond_likelihood, axis = 1) # int index of horrizontal max
    pred_labels = max_idx.astype('float64')
    # Compute and return the most likely class
    return pred_labels

def pred_accuracy(pred_labels, true_labels):
    N = pred_labels.shape[0]
    pred_labels = pred_labels.astype(int)
    true_labels = true_labels.astype(int)
    match = np.sum(np.ones(N)[pred_labels==true_labels])
    accuracy = float(match)/N
    return accuracy

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # show the diagonal elements
    plot_cov_diagonal(covariances)

    # Evaluation
    ## Average conditional likelihoods
    train_avg_cond_likelihood = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_cond_likelihood = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print('Average conditional log-likelihood on train set is: {0}\
         \nAverage conditional log-likelihood on test set is: {1}'\
          .format(train_avg_cond_likelihood,test_avg_cond_likelihood))

    ## Predict most likely posterior class, and report accuracy
    train_pred = classify_data(train_data, means, covariances)
    test_pred = classify_data(test_data, means, covariances)
    train_accuracy = pred_accuracy(train_pred, train_labels)
    test_accuracy = pred_accuracy(test_pred, test_labels)
    print('Accuracy on train set is : {0}\
         \nAccuracy on test set is : {1}'.format(train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()