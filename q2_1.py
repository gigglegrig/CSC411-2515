'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data  # n*64
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

#     def fit(self, train_data, train_labels):
#         self.train_data = train_data
#         self.train_labels = train_labels
#         self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)

        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)

        # print(train_norm.shape, self.train_norm.shape, self.train_data.shape, test_point.shape)

        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose()) # (X - X')^2, shape [n*1]
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
#         digit = None
        l2_dist = self.l2_distance(test_point)     # n*1 vector of L2 dist
        top_k_idx = np.argsort(l2_dist)[:k]  # idx of k nn
        top_k_label = self.train_labels[top_k_idx] # label of k nn

#         label_counter = {}  # label:count
#         for label in top_k_label:
#             label_counter[label] = label_counter.get(label, 0) + 1 # if none, get returns 0, else ++
#         # break tie by choosing the smaller label alphabetically
#         digit_label = sorted(label_counter.items(), key=lambda x: (-x[1],-x[0]))[0][0] #descending value, ascending key
        ####
        label_dist_counter = {} # label:(count, total_dist)
        for label in set(top_k_label):
            ones = np.ones(k)
            label_idx = top_k_idx[top_k_label==label]
            total_dist = np.sum(l2_dist[label_idx])
            label_count = np.sum(ones[top_k_label==label])
            label_dist_counter[label] = (label_count, total_dist) # k:(count, dist)
        ####
        digit_label = sorted(label_dist_counter.items(), key=lambda x: (-x[1][0],x[1][1],x[0]))[0][0]
        return digit_label

def cross_validation(knn, k_range=np.arange(1,15)):
    avg_val = []
    train_data = knn.train_data
    train_labels = knn.train_labels
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        kf = KFold(n_splits=10, shuffle=False)
        val_accuracy = np.ones(10)
        for i,(train, test) in enumerate(kf.split(train_data)):
            temp_knn = KNearestNeighbor(train_data[train], train_labels[train])
            eval_data = train_data[test]
            eval_labels = train_labels[test]
            val_accuracy[i] = classification_accuracy(temp_knn, k, eval_data, eval_labels)
        avg_val.append(np.mean(val_accuracy))
    return avg_val


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    pred_label = []
    for i in range(len(eval_data)):
        pred_label.append(knn.query_knn(eval_data[i], k))
    match = sum(np.ones(len(eval_labels))[np.array(eval_labels) == np.array(pred_label)])
    accuracy = float(match)/len(eval_labels)
    return accuracy

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data', shuffle=True)
    knn = KNearestNeighbor(train_data, train_labels)
    # accuracy for k=1
    train_accuracy1 =  classification_accuracy(knn, 1, train_data, train_labels)
    test_accuracy1 = classification_accuracy(knn, 1, test_data, test_labels)
    print("Train accuracy for K=1: {}\
         \nTest accuracy for K=1: {}".format(train_accuracy1,test_accuracy1))
    # k=15
    train_accuracy15 =  classification_accuracy(knn, 15, train_data, train_labels)
    test_accuracy15 = classification_accuracy(knn, 15, test_data, test_labels)
    print("Train accuracy for K=15: {}\
         \nTest accuracy for K=15: {}".format(train_accuracy15,test_accuracy15))
    # cross validation to choose optimal k
    val_score = cross_validation(knn)
    print("Val_scores on each fold : {0}".format(val_score))
    opt_k = np.arange(1,15)[np.argmax(val_score)]
    print("Optimal hyperparameter K is : {}".format(opt_k))
    # accuracy for optimal k
    train_accuracy =  classification_accuracy(knn, opt_k, train_data, train_labels)
    test_accuracy = classification_accuracy(knn, opt_k, test_data, test_labels)
    print('Train classification accuracy: {0} \
         \nAverage Accuracy across folds: {1} \
         \nTest Accuracy: {2}'.format(train_accuracy,val_score[opt_k-1],test_accuracy))

if __name__ == '__main__':
    main()