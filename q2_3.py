'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    # eta with prior
    eta = np.ones((10, 64))
    one_array = np.ones(train_labels.shape)
    for label in set(train_labels):
        N_k = np.sum(one_array[train_labels == label]) + 2 # scalar, # of class k, + a + b
        N_kj = np.sum(train_data[train_labels == label], axis=0) + 1
        eta[int(label)] = N_kj / N_k
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    imgs = []
    for i in range(10):
        img_i = class_images[i]
        imgs.append(np.reshape(img_i, (8,8)))
    concat = np.concatenate(imgs, 1)
    plt.imshow(concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
#     generated_data = np.zeros((10, 64))
    generated_data = binarize_data(eta)
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''
    bin_digits = np.reshape(bin_digits, (-1,64))
    # Loop version
#     N = bin_digits.shape[0]
#     D = 64
#     gen_likelihood = np.zeros((N,10))
#     for i,digit in enumerate(bin_digits):
#         likelihoods = np.zeros(10)
#         for k in range(10): # p(x|y=k) for each of k in [0,9]
#             w_k = np.log(eta[k]/(1-eta[k])) # 1*64
#             w0_k = np.sum(np.log(1-eta[k])) # scalar
#             likelihoods[k] = w_k.dot(digit.T) + w0_k
#         gen_likelihood[i] = likelihoods
#     # vectorization
    w_k = np.log(eta/(1-eta))
    w0_k = np.sum(np.log(1-eta.T),axis=0)
    gen_likelihood = bin_digits.dot(w_k.T) + w0_k[np.newaxis, :]
    return gen_likelihood

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    bin_digits = np.reshape(bin_digits, (-1,64))
    # loop version
#     N = digits.shape[0]
#     cond_likelihood = np.zeros((N,10))
#     for i,digit in enumerate(bin_digits):
#         likelihoods = np.zeros((N,10))
#         for k in range(10):
#             w_k = np.log(eta[k]/(1-eta[k])) # 1*64
#             b_k = np.sum(np.log(1-eta[k])) + np.log(1/10) # scalar
#             likelihoods[k] = w_k.dot(digit.T) + b_k
    # vectorizing
    gen_likelihood = generative_likelihood(bin_digits, eta)
    log_total_prob = np.log(np.sum(np.exp(gen_likelihood), axis=1))
    cond_likelihood =gen_likelihood - log_total_prob[:, np.newaxis]
    return cond_likelihood

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Output the likelihood for distinct true labels, the average over all labels is done in main()

    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits,eta)
    N = bin_digits.shape[0]
    total = 0
    for n in range(N):
        total = total + cond_likelihood[n][int(labels[n])]
    average = total/N
    return average

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    max_idx = np.argmax(cond_likelihood, axis=1)
    pred_labels = max_idx.astype('float64')
    # Output the likelihood for distinct true labels, the average over all labels is done in main()

    return pred_labels

def pred_accuracy(pred_labels, true_labels):
    '''
    Compare the predicted labels and true labels, return the percentage of accuracy
    Input: Predict_labels, vector. True_labels, vector
    Output: percentage representing the accuracy
    '''
    N = pred_labels.shape[0]
    pred_labels = pred_labels.astype(int)
    true_labels = true_labels.astype(int)
    match = np.sum(np.ones(N)[pred_labels==true_labels])
    accuracy = float(match)/N
    return accuracy

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)
    ## Avg conditonal likelihood
    train_avg_cond_likelihood = avg_conditional_likelihood(train_data, train_labels, eta)
    test_avg_cond_likelihood = avg_conditional_likelihood(test_data, test_labels, eta)
    print('Average conditional log-likelihood on train set is: {0}\
         \nAverage conditional log-likelihood on test set is: {1}'\
          .format(train_avg_cond_likelihood,test_avg_cond_likelihood))

    ## Predict most likely posterior class, and report accuracy
    train_pred = classify_data(train_data, eta)
    test_pred = classify_data(test_data, eta)
    train_accuracy = pred_accuracy(train_pred, train_labels)
    test_accuracy = pred_accuracy(test_pred, test_labels)
    print('Accuracy on train set is : {0}\
         \nAccuracy on test set is : {1}'.format(train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()