from unittest import TestCase

from fuzzywuzzy import fuzz

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split

# Demonstrate K-neighbor classification(K近邻算法) ML:
#  https://blog.csdn.net/suipingsp/article/details/41964713
#  https://blog.csdn.net/u011067360/article/details/23941577
#
from sklearn.neighbors import KNeighborsClassifier

class TestSciKit(TestCase):
    def test_knn(self):
        # Get Iris data and target
        iris_X, iris_y = datasets.load_iris(return_X_y=True)

        # Split test data. train_test_split() will not only split data, but also randomly re-sorts the order of the data.
        X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

        # Use classification (分类学习法) to train our (test) data
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)    # Training!!!

        score = knn.predict(X_test)  # predict
        print()
        print(score)
        print(y_test)

        # assert
        ratio = fuzz.partial_ratio(score, y_test)
        print("Ratio: " + str(ratio))
        assert(ratio >= 95)
