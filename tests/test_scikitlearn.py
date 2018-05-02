from unittest import TestCase

from fuzzywuzzy import fuzz

#import numpy as np
from sklearn import datasets

class TestSciKit(TestCase):
    def test_knn(self):
        '''
        Get Iris data and target.  X 是每朵花的数据，y 是对应的每朵花的分类.
        SciKit 样本库提供了 150 朵 iris 花的模拟数据，并预先分好了类。
        关于该库的官方说明: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris

        K-nearnest neighbor classification(K近邻算法): 
            根据已知的邻近(相似)数据(X)的分类(y)推算出当前数据的分类。属于分类 (classification) 算法

        参考:
            * https://coolshell.cn/articles/8052.html
            * https://blog.csdn.net/suipingsp/article/details/41964713
            * http://www.cnblogs.com/daniel-D/p/3244718.html
            * https://blog.csdn.net/u011067360/article/details/45937327
        '''
        # Demonstrate K-neighbor classification(K近邻算法) ML:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.cross_validation import train_test_split

        iris_X, iris_y = datasets.load_iris(return_X_y=True)

        # Split test data. train_test_split() will not only split data, but also randomly re-sorts the order of the data.
        train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size=0.3)

        # 用 KNN classification (分类学习法) to train our (test) data. 假设 K 取 5(这其实也是缺省值)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_X, train_y)    # Training!!!

        score = knn.predict(test_X)  # predict
        print("\n" + str(score))
        print(test_y)

        # test: compare score output and actual result.
        ratio = fuzz.partial_ratio(score, test_y)
        print("Ratio: " + str(ratio))
        assert(ratio >= 95)

    def test_linear_regression(self):
        '''
        SciKit learn 提供了模拟数据 API

        Linear regression 
            根据已知结果(y)和影响结果的因数(X)，推算出新数据的结果。
        '''
        from sklearn.linear_model import LinearRegression
        from sklearn.cross_validation import train_test_split

        boston = datasets.load_boston()   # By default(without 'return_X_y=True' argument), it will return a Bunch set
        data_X = boston.data
        target_y = boston.target

        # Boston dataset totally has 506 records, we will only use 5(1%) of them for test
        #train_X, test_X, train_y, test_y = train_test_split(data_X, target_y, test_size=0.01)  
        train_X = data_X[:-5]      # from head until last 5 records
        train_y = target_y[:-5]
        test_X = data_X[-5:]       # last 5 records
        test_y = target_y[-5:]

        model = LinearRegression()
        model.fit(train_X, train_y)      # training

        # coefficient: 决定系数. 对于简单线性回归而言，决定系数为样本"相关系数(Correlation)"的平方. 相关系数显示两个随机变量之间线性关系的强度和方向
        # coefficient 的值的计算过程：
        #   1) 将每项数据减去其预测值得到每项数据的“残差”
        #   2) 计算出每项数据“残差”的平方总和 SS_res
        #   3) 计算出数据的平均值
        #   4) 将每项数据减去平均值，然后计算出他们的平方和，称为“总平方和” SS_tot
        #   5) 1 减去 SS_res/SS_tot 就得到决定系数
        #   6) 决定系数的开方可以得到样本的"相关系数(Correlation)"
        #   参考：https://zh.wikipedia.org/wiki/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0
        print(model.coef_)

        # Intercept: 截距是在 x=0 的时候求得的y值，它的含义是去除输入后的预测模型。可以理解为是一元回归中任何一个有意义的预测值的起点。
        print(model.intercept_)

        score = model.predict(test_X)    # predict with test data
        print("\n" + str(score))
        print(test_y)

    def test_mock_data(self):
        '''
        SciKit learn 提供了模拟数据API
        '''
        import matplotlib.pyplot as plot

        # 生成 100 个模拟数据：
        #  n_features： 生成的数据的 columns 数
        #  n_targets：  生成的模拟结果的 columns 数
        #  noise：      增加点间的离散度
        X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=5)

        # 可视化展示数据
        plot.scatter(X, y)   # 显示方式为“离散化”，也就是以“点”的方式展示数据关系
        plot.show()

    def test_model(self):
        pass