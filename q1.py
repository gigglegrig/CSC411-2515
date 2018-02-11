from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    # iterate from 1st subplot to the 15th
    for i in range(feature_count):
        ax= plt.subplot(3, 5, i + 1)
        # TODO: Plot feature i against y
        plt.scatter(X[:,i],y, s=1)
        ax.set_xlabel(features[i])
        ax.set_ylabel('Median Value')
    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    # TODO: implement linear regression
    X_biased = np.column_stack((np.ones(X.shape[0]), X))
    w = np.linalg.solve(X_biased.T.dot(X_biased), X_biased.T.dot(Y))

    # Remember to use np.linalg.solve instead of inverting!
    # raise NotImplementedError()
    return w


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    print("Shape of X: {}\nShape of y: {}".format(X.shape, y.shape))
    # Visualize the features
    visualize(X, y, features)

    # TODO: Split data into train and test
    train_size = round(X.shape[0] * 0.8)
    test_size = X.shape[0] - train_size

    random_train_idx = np.random.choice(506, train_size, replace=False)
    random_test_idx = [x for x in range(506) if x not in random_train_idx]
    X_train = X[random_train_idx]
    y_train = y[random_train_idx]
    X_test = X[random_test_idx]
    y_test = y[random_test_idx]

    # Fit regression model
    w = fit_regression(X_train, y_train)
    X_test_biased = np.column_stack((np.ones(X_test.shape[0]), X_test))
    # Compute fitted values, MSE, etc.
    MSE = np.mean((X_test_biased.dot(w) - y_test) ** 2)
    print("MSE = {}".format(MSE))
    # Alternative error measurement metrics
    # RMSE = np.sqrt(np.mean((X_test_biased.dot(w) - y_test) ** 2))
    # print("RMSE = {}".format(RMSE))
    # MAE = np.mean(np.absolute(X_test_biased.dot(w) - y_test))
    # print("MAE = {}".format(MAE))

if __name__ == "__main__":
    main()

