'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint as sp_randint
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    # import and filter data

    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),shuffle=True)
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),shuffle=True)

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer(stop_words="english")
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def model_train(model, train, train_labels, test, test_labels):
    if model =='lr':
        clf = LogisticRegression()
        param_dist = {'C': [0.001, 0.01, 0.1, 1]}
    elif model == 'mn':
        clf = MultinomialNB()
        param_dist = {'alpha':[0.001, 0.01, 0.1, 1]}
    elif model == 'rf':
        clf = RandomForestClassifier(n_estimators=200,max_features='auto')
        param_dist = {'max_depth':[10,30,50]}
    else:
        print('Choose from lr, mn, rf.')
    inner_cv = StratifiedKFold(random_state=1)
    outer_cv = StratifiedKFold(random_state=2)
    # Inner loop: grid search to find opt hyperprameter
    grid = GridSearchCV(estimator=clf, param_grid=param_dist, cv=inner_cv)
    grid.fit(train,train_labels)
    # Outter loop: cross validation with the chosen hyperparameters
    nested_score = cross_val_score(grid, X=train,y=train_labels, cv=outer_cv)
    # obtain mean CV scores.
    train_score = nested_score.mean()
    # obtain test score
    test_score = (grid.predict(test)==test_labels).mean()
    # return train/test score
    model_fullname = {'mn':'MultinomialNB', 'rf':'RandomForestClassifier', 'lr':'LogisticRegression'}
    print('For model: {}, train_score:{}, test_score:{}'.format(model_fullname[model],train_score,test_score))

    return train_score,test_score, grid

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_tfidf, test_tfidf, feature_names = tf_idf_features(train_data, test_data)
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    mn_train,mn_test,mn_model = model_train('mn',train_tfidf, train_data.target, test_tfidf, test_data.target)
    rf_train,rf_test,rf_model = model_train('rf',train_tfidf, train_data.target, test_tfidf, test_data.target)
    lr_train,lr_test,lr_model = model_train('lr',train_bow, train_data.target, test_bow, test_data.target)

    print('Confusion Matrix and the corresponding heatmap of MultinomialNB are shown below')
    cm = np.zeros((20,20))
    for i,j in zip(mn_model.predict(test_tfidf), test_data.target):
        cm[i][j] += 1
    print(cm)
    sns.heatmap(cm)
    sns.plt.show()