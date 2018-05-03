from unittest import TestCase

class NormalizationTest(TestCase):
    def test_normalization(self):
        '''
          Normalization 又称为 scale，它的作用是让数据更集中在设定的范围内。在 ML 中数据的取值区域如果太宽会导致内涵太丰富以至于无法解释
          因此需要将其转化为无量纲的纯数值以便于比较。其中最典型的是归一化处理，即将数据映射到[0，1]区间。

          原理解释：
            https://zhuanlan.zhihu.com/p/33173246
            http://www.360doc.com/content/18/0122/22/7378868_724269731.shtml

          Python 示例：
            https://blog.csdn.net/pipisorry/article/details/52247679
        '''

        from sklearn import preprocessing
        import numpy as np

        a = np.array([[ 10,  2.7, 3.6],
                     [-100, 5,  -2],
                     [ 120, 20,  40]], dtype=np.float64)
        # 以上三组数据每组对应位置的值，比如 A1：B1：C1 分别为 10, -100, 120，分布在 超过200的区间，scale后的结果会大大地缩小这个跨度。
        nor_a = preprocessing.scale(a)
        print(nor_a)

    def test_SVC_classfication(self):
      '''
      SVM：支持向量机。用于数据分类, 支持多元分类。适用于监督学习的分类和线形回归.
      SVC：C-Support Vector Classification.

      什么是 SVM:
        http://www.cnblogs.com/LeftNotEasy/archive/2011/05/02/basic-of-svm.html

      SciKitLearn 支持的 SVM 算法:
        http://sklearn.apachecn.org/cn/0.19.0/modules/svm.html
        https://xacecask2.gitbooks.io/scikit-learn-user-guide-chinese-version/content/sec1.4.html
        https://blog.csdn.net/gamer_gyt/article/details/51265347 
      '''
      from sklearn.svm import SVC

      # Tools
      from sklearn import preprocessing
      import numpy as np
      from sklearn.cross_validation import train_test_split

      # For mockup classification data
      from sklearn.datasets.samples_generator import make_classification

      # For virtualizaiton
      import matplotlib.pyplot as plot

      # Mock up data
      X, y = make_classification(n_samples=300,
                                 n_features=2,      # 两个 feature
                                 n_redundant=0,
                                 n_informative=2,
                                 random_state=22,   # Random seed，固定数值导致每次输出相同(纯函数特性)
                                 n_clusters_per_class=1,
                                 scale=100)
      
      X = preprocessing.minmax_scale(X, feature_range=(-1, 1))
      # plot.scatter(X[:,0], X[:,1], c=y)   # 以两个 Feature 分别作为坐标轴
      # plot.show()

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
      clf = SVC()
      clf.fit(X_train, y_train)

      print( clf.score(X_test, y_test) )      # 0.93. 如果不做 Normalization，score 的结果只有 0.45 左右

