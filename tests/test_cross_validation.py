from unittest import TestCase

class CrossValidationTest(TestCase):
    def test_cross_validation_1(self):
        '''
        交叉验证方法一：使用相同数据集划分出不同的 testing dataset 来多次检验同一个模型并给出平均得分
        '''
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.cross_validation import train_test_split
        from sklearn import datasets

        iris_X, iris_y = datasets.load_iris(return_X_y=True)

        k_range = range(1, 30)
        k_scores = []
        k_losses = []
        for k in k_range:    # 测试 30 种 knn 的临近距离，观察哪个距离最好。
            knn = KNeighborsClassifier(n_neighbors=k)

            from sklearn.cross_validation import cross_val_score
            scores = cross_val_score(knn,
                                    iris_X, iris_y,        # 测试数据全集
                                    cv=10,                 # 自动将测试数据切分出 10 组 test data
                                    scoring='accuracy')    # 'accuracy' for Classification.
            k_scores.append(scores.mean())    # 求平均(10 次 cv 得到 10 个结果)

            losses = -cross_val_score(knn, iris_X, iris_y, cv=10, scoring='neg_mean_squared_error')  # for Regression，通过方差计算损失(是个负数，因此结果要再次负回来，结果越小越好)
            k_losses.append(losses.mean())
        # 可视化输出
        from matplotlib import pyplot as plt
        plt.plot(k_range, k_scores, color="b")
        plt.plot(k_range, k_losses, color="g")
        plt.xlabel("n_neighbors")
        plt.ylabel("scores/k_losses")
        plt.show()

    def test_learning_curve(self):
        '''
        交叉验证：可视化比较 training 和 testing 的结果
        '''
        from sklearn.datasets import load_digits
        from sklearn.svm import SVC
        import numpy as np

        from matplotlib.pyplot import plot

        digits = load_digits()   # digits 是一个二维数组, 共有 1797 组数据，每组 64 个数字
        X = digits.data
        y = digits.target

        '''
        SVC：虚拟向量机应用于分类学习。
        
        learning_curve 通过交叉验证(cv)，返回 training 的时候的损失值和 test 的时候的损失值，称为学习曲线。
        '''
        from sklearn.learning_curve import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            SVC(gamma=0.001),
            X, y,
            cv=10,                                   # 分 10 组，也就是每组大约占全部训练数据的 10%
            scoring='neg_mean_squared_error',
            train_sizes=[0.1, 0.25, 0.5, 0.75, 1]    # 分别取训练数据总数( 90%，要减去 test 的10% )的 0.1(10%), 0.25, 0.5, 0.75 和 100% 作为训练集，
                                                     # 因为有5组，因此也会得到5组结果(train_loss, test_loss)。每组因为交叉训练(cv)10次，因此得到一个
                                                     # 包含 10 个结果的数组。因此 train_loss 和 test_loss 都是一个 10x5 的数组
        )

        train_loss_mean = -np.mean(train_scores, axis=1)
        test_loss_mean = -np.mean(test_scores, axis=1)

        # 可视化输出
        from matplotlib import pyplot as plt
        plt.plot(train_sizes, train_loss_mean, 'o-', color="b", label="Training loss")  # Traning 的曲线比较平直，因为模型的算法会尽可能匹配训练数据，因此会呈现出“自匹配”的结果，但是这个结果是不是真的好，则要看 testing 的结果
        plt.plot(train_sizes, test_loss_mean, 'o-', color="r", label="Testing loss")    # Testing 的曲线随着 training 数据集的增大下降较快，因为越大的数据集对最终的结果越好。
                                                                                        # 两条图线的比较可以看到 training 需要 test 来参照才能判断出训练效果。
                                                                                        # 如果两条曲线不是呈收敛状态，training 持续下降，而testing则上升，那很可能就是 over fitting 或 under fitting。
                                                                                        # 通过调整 SVC 的 gamma 参数来调整训练效果。

        plt.xlabel("Training sizes")
        plt.ylabel("Loss")
        plt.show()

    def test_over_fitting(self):
        from sklearn.datasets import load_digits
        from sklearn.svm import SVC
        import numpy as np

        from matplotlib.pyplot import plot

        digits = load_digits()
        X = digits.data
        y = digits.target

        '''
        validation_curve 允许我们动态测试算法参数以寻找过拟合拐点。
        和 learning_curve 不同的是，learning_curve 展示的是数据集的学习曲线，因此允许划分数据集大小，validation_curve 则没有这个参数，
        因此不需要 train_sizes 参数也不会返回 train_sizes actual value. 直接返回各个（5个）参数下的 train_score 和 test_score
        '''
        from sklearn.learning_curve import validation_curve
        svc_gamma_range = np.logspace(-5, -2, 5)             # 从 power(10, -5) 到 power(10, -2) 之间(0.00001~0.01)随机取 5 个点
        train_scores, test_scores = validation_curve(
            SVC(),
            X, y,
            param_name='gamma', param_range=svc_gamma_range,   # 动态修改 SVC 的 gamma 参数值
            cv=10,
            scoring='neg_mean_squared_error'
        )
        train_loss_mean = -np.mean(train_scores, axis=1)
        test_loss_mean = -np.mean(test_scores, axis=1)

        # 可视化输出
        from matplotlib import pyplot as plt
        plt.plot(svc_gamma_range, train_loss_mean, 'o-', color="b", label="Training loss") # 蓝线反映训练结果
        plt.plot(svc_gamma_range, test_loss_mean, 'o-', color="r", label="Testing loss")   # 红线反映测试结果，在某个 gamma 值之后出现拐点，和训练结果背道而驰。

        plt.xlabel("SVC gamma value")
        plt.ylabel("Loss")
        plt.show()
