import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


class Classifier:

    def __init__(self, k=100):
        self.testing_X = pd.read_csv('../data/Test/X_test.txt', sep=" ", header=None)
        self.testing_y = pd.read_csv('../data/Test/y_test.txt', sep=" ", header=None)
        self.testing_si = pd.read_csv('../data/Test/subject_id_test.txt', sep=" ", header=None)

        self.training_X = pd.read_csv('../data/Train/X_train.txt', sep=" ", header=None)
        self.training_y = pd.read_csv('../data/Train/y_train.txt', sep=" ", header=None)
        self.training_si = pd.read_csv('../data/Train/subject_id_train.txt', sep=" ", header=None)
        k_indices = np.random.choice(self.training_y.shape[0], int(((100 - k) / 100) * self.training_y.shape[0]),
                                     replace=False)

        self.training_X = self.training_X.drop(labels=k_indices, axis=0)
        self.training_y = self.training_y.drop(labels=k_indices, axis=0)
        self.training_si = self.training_si.drop(labels=k_indices, axis=0)

        indices = np.random.choice(self.training_y.shape[0], int(0.1 * self.training_y.shape[0]), replace=False)
        self.training_X = self.training_X.reset_index(drop=True)
        self.training_y = self.training_y.reset_index(drop=True)
        self.training_si = self.training_si.reset_index(drop=True)

        self.validation_X = self.training_X.take(indices, axis=0)
        self.validation_y = self.training_y.take(indices, axis=0)
        self.validation_si = self.training_si.take(indices, axis=0)

        self.training_X = self.training_X.drop(labels=indices, axis=0)
        self.training_y = self.training_y.drop(labels=indices, axis=0)
        self.training_si = self.training_si.drop(labels=indices, axis=0)

    def logistic_regression_fit(self, cross_validation=True):
        max_iters = [80, 100, 150, 200]
        solvers = ['lbfgs', 'liblinear']
        if not cross_validation:
            solvers = ['lbfgs']
            max_iters = [150]
        best_acc = 0
        best_clf = None
        for sol in solvers:
            for iter in max_iters:
                clf = LogisticRegression(random_state=0, multi_class='auto', max_iter=iter, solver=sol, C=np.inf).fit(
                    self.training_X, self.training_y)
                y_hat = clf.predict(self.validation_X).reshape((self.validation_y.shape[0], 1))
                if (np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]) > best_acc:
                    best_acc = np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]
                    best_clf = clf
        self.lr_classifier = best_clf

    def logistic_regression_predict(self):
        return self.lr_classifier.predict(self.testing_X).reshape((self.testing_y.shape[0], 1))

    def logistic_regression_accuracy(self):
        y_hat = self.logistic_regression_predict()
        print('Logistic Regression Accuracy:')
        print(np.sum(y_hat == np.array(self.testing_y)) / self.testing_y.shape[0])

    def random_forest_fit(self, cross_validation=True):
        trees = [3, 10, 15, 20]
        attrs = ['sqrt', 'log2']
        depths = [5, None]
        if not cross_validation:
            depths = [None]
            attrs = ['sqrt']
            trees = [15]
        best_acc = 0
        best_clf = None
        for t in trees:
            for a in attrs:
                for d in depths:
                    clf = RandomForestClassifier(n_estimators=t, max_depth=d, random_state=0, max_features=a).fit(
                        self.training_X, self.training_y)
                    y_hat = clf.predict(self.validation_X).reshape((self.validation_y.shape[0], 1))
                    if (np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]) > best_acc:
                        best_acc = np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]
                        best_clf = clf
        self.rf_classifier = best_clf

    def random_forest_predict(self):
        return self.rf_classifier.predict(self.testing_X).reshape((self.testing_y.shape[0], 1))

    def random_forest_accuracy(self):
        y_hat = self.random_forest_predict()
        print('Random Forest Accuracy:')
        print(np.sum(y_hat == np.array(self.testing_y)) / self.testing_y.shape[0])

    def svm_fit(self, cross_validation=True):
        kernels = ['linear', 'poly', 'rbf']
        ds = [2, 3, 4]  # this degree will affect the classifier only when the kernel is 'poly'
        dec = ['ovo', 'ovr']
        if not cross_validation:
            kernels = ['poly']
            ds = [2]
            dec = ['ovo']
        best_acc = 0
        best_clf = None
        for k in kernels:
            for d in dec:
                if k == 'poly':
                    for degree in ds:
                        clf = SVC(gamma='scale', decision_function_shape=d, kernel=k, degree=degree, C=np.inf,
                                  class_weight='balanced').fit(self.training_X, self.training_y)
                        y_hat = clf.predict(self.validation_X).reshape((self.validation_y.shape[0], 1))
                        if (np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]) > best_acc:
                            best_acc = np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]
                            best_clf = clf
                else:
                    clf = SVC(gamma='scale', decision_function_shape=d, kernel=k, class_weight='balanced',
                              C=np.inf).fit(self.training_X, self.training_y)
                    y_hat = clf.predict(self.validation_X).reshape((self.validation_y.shape[0], 1))
                    if (np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]) > best_acc:
                        best_acc = np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]
                        best_clf = clf
        self.svm_classifier = best_clf

    def svm_predict(self):
        return self.svm_classifier.predict(self.testing_X).reshape((self.testing_y.shape[0], 1))

    def svm_accuracy(self):
        y_hat = self.svm_predict()
        print('SVM Accuracy:')
        print(np.sum(y_hat == np.array(self.testing_y)) / self.testing_y.shape[0])

    def adaboost_fit(self, cross_validation=True):
        depths = [1, 2, 5]
        trees = [20, 50, 70, 100]
        if not cross_validation:
            depths = [5]
            trees = [70]
        best_clf = None
        best_acc = 0
        for d in depths:
            for t in trees:
                clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=d), n_estimators=t, learning_rate=1.5,
                                         algorithm="SAMME").fit(self.training_X, self.training_y)
                y_hat = clf.predict(self.validation_X).reshape((self.validation_y.shape[0], 1))
                if (np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]) > best_acc:
                    best_acc = np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]
                    best_clf = clf
        self.ab_classifier = best_clf

    def adaboost_predict(self):
        return self.ab_classifier.predict(self.testing_X).reshape((self.testing_y.shape[0], 1))

    def adaboost_accuracy(self):
        y_hat = self.adaboost_predict()
        print('AdaBoost Accuracy:')
        print(np.sum(y_hat == np.array(self.testing_y)) / self.testing_y.shape[0])

    def neural_net_fit(self, cross_validation=True):
        hiddens = [(50,), (100,), (300,)]
        activations = ['logistic', 'relu']
        solvers = ['lbfgs', 'adam']
        if not cross_validation:
            hiddens = [(100,)]
            activations = ['relu']
            solvers = ['adam']
        best_clf = None
        best_acc = 0
        for h in hiddens:
            for a in activations:
                for s in solvers:
                    clf = MLPClassifier(hidden_layer_sizes=h, activation=a, solver=s, alpha=0).fit(self.training_X,
                                                                                                   self.training_y)
                    y_hat = clf.predict(self.validation_X).reshape((self.validation_y.shape[0], 1))
                    if (np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]) > best_acc:
                        best_acc = np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]
                        best_clf = clf
        self.nn_classifier = best_clf

    def neural_net_predict(self):
        return self.nn_classifier.predict(self.testing_X).reshape((self.testing_y.shape[0], 1))

    def neural_net_accuracy(self):
        y_hat = self.neural_net_predict()
        print('Neural Net Accuracy:')
        print(np.sum(y_hat == np.array(self.testing_y)) / self.testing_y.shape[0])

    def linear_svm_loss_fit(self):
        coefs = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1, 1, 10, 100]
        penalties = ['l1', 'l2']
        self.svm_classifier_loss = {}
        for p in penalties:
            print('Penalty', p)
            best_clf = None
            best_acc = 0
            for co in coefs:
                c = 1 / co
                clf = LinearSVC(random_state=0, penalty=p, C=c, dual=False).fit(self.training_X, self.training_y)
                y_hat = clf.predict(self.validation_X).reshape((self.validation_y.shape[0], 1))
                print(np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0])
                if (np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]) > best_acc:
                    best_acc = np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]
                    best_clf = clf
                print('coef: ', co, '\tnonzero weights:', len(np.where(np.abs(clf.coef_) > 1e-6)[0]))
            self.svm_classifier_loss[p] = best_clf

    def linear_svm_loss_predict(self):
        return {'l1': self.svm_classifier_loss['l1'].predict(self.testing_X).reshape((self.testing_y.shape[0], 1)),
                'l2': self.svm_classifier_loss['l2'].predict(self.testing_X).reshape((self.testing_y.shape[0], 1))}

    def linear_svm_loss_accuracy(self):
        predictions = self.linear_svm_loss_predict()
        # print('LinearSVM with Loss Accuracy:')
        # for p, y_hat in predictions.items():
        #     print(p + ' penalty:')
        #     print(np.sum(y_hat == np.array(self.testing_y)) / self.testing_y.shape[0])

    def logistic_regression_loss_fit(self):
        coefs = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1, 1, 10, 100]
        penalties = ['l1', 'l2']
        self.lr_classifier_loss = {}
        for p in penalties:
            print('Penalty', p)
            best_clf = None
            best_acc = 0
            for co in coefs:
                c = 1 / co
                clf = LogisticRegression(random_state=0, multi_class='auto', penalty=p, C=c).fit(self.training_X,
                                                                                                 self.training_y)
                y_hat = clf.predict(self.validation_X).reshape((self.validation_y.shape[0], 1))
                print(np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0])
                if (np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]) > best_acc:
                    best_acc = np.sum(y_hat == np.array(self.validation_y)) / self.validation_y.shape[0]
                    best_clf = clf
                print('coef: ', co, '\tnonzero weights:', len(np.where(np.abs(clf.coef_) > 1e-6)[0]))
            self.lr_classifier_loss[p] = best_clf

    def logistic_regression_loss_predict(self):
        return {'l1': self.lr_classifier_loss['l1'].predict(self.testing_X).reshape((self.testing_y.shape[0], 1)),
                'l2': self.lr_classifier_loss['l2'].predict(self.testing_X).reshape((self.testing_y.shape[0], 1))}

    def logistic_regression_loss_accuracy(self):
        predictions = self.logistic_regression_loss_predict()
        # print('Logistic Regression with Loss Accuracy:')
        # for p, y_hat in predictions.items():
        #     print(p + ' penalty:')
        #     print(np.sum(y_hat == np.array(self.testing_y)) / self.testing_y.shape[0])

    def feature_selection(self):
        previous_training_X = self.training_X.copy(deep=True)
        previous_testing_X = self.testing_X.copy(deep=True)
        previous_validation_X = self.validation_X.copy(deep=True)

        f = previous_testing_X.shape[0]
        for l in [561, 100, 50, 10, 5]:
            print('Feature', l)
            clf = ExtraTreesClassifier(n_estimators=10).fit(previous_training_X, self.training_y)
            model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=l)
            self.training_X = model.transform(previous_training_X)
            self.testing_X = model.transform(previous_testing_X)
            self.validation_X = model.transform(previous_validation_X)

            # classifier.logistic_regression_fit(cross_validation=False)
            # classifier.logistic_regression_accuracy()
            #
            # classifier.random_forest_fit(cross_validation=False)
            # classifier.random_forest_accuracy()
            #
            classifier.svm_fit(cross_validation=False)
            classifier.svm_accuracy()
            #
            # classifier.adaboost_fit(cross_validation=False)
            # classifier.adaboost_accuracy()

        self.training_X = previous_training_X
        self.testing_X = previous_testing_X
        self.validation_X = previous_validation_X

    def best_feature_method(self):
        previous_training_X = self.training_X.copy(deep=True)
        previous_testing_X = self.testing_X.copy(deep=True)
        previous_validation_X = self.validation_X.copy(deep=True)

        for p in ['l1', 'l2']:
            lsvc = LinearSVC(C=0.01, penalty=p, dual=False).fit(previous_training_X, self.training_y)
            model = SelectFromModel(lsvc, prefit=True, max_features=10, threshold=-np.inf)
            self.training_X = model.transform(previous_training_X)
            self.validation_X = model.transform(previous_validation_X)
            self.testing_X = model.transform(previous_testing_X)

            classifier.logistic_regression_fit(cross_validation=False)
            classifier.logistic_regression_accuracy()


classifier = Classifier()


def q_1():
    # classifier.logistic_regression_fit()
    # classifier.logistic_regression_accuracy()

    # classifier.random_forest_fit()
    # classifier.random_forest_accuracy()

    classifier.svm_fit()
    classifier.svm_accuracy()
    #
    # classifier.adaboost_fit()
    # classifier.adaboost_accuracy()
    #
    # classifier.neural_net_fit()
    # classifier.neural_net_accuracy()


def q_2():
    classifier.linear_svm_loss_fit()
    classifier.linear_svm_loss_accuracy()

    # classifier.logistic_regression_loss_fit()
    # classifier.logistic_regression_loss_accuracy()


def q_3():
    for k in [5, 10, 20, 50, 100]:
        print('Proportion', k, '% :')
        classifier = Classifier(k=k)
        classifier.logistic_regression_fit(cross_validation=False)
        classifier.logistic_regression_accuracy()
    print('=' * 55)
    for k in [5, 10, 20, 50, 100]:
        print('Proportion', k, '% :')
        classifier = Classifier(k=k)
        classifier.random_forest_fit(cross_validation=False)
        classifier.random_forest_accuracy()
    print('=' * 55)
    for k in [5, 10, 20, 50, 100]:
        print('Proportion', k, '% :')
        classifier = Classifier(k=k)
        classifier.svm_fit(cross_validation=False)
        classifier.svm_accuracy()
    print('=' * 55)
    for k in [5, 10, 20, 50, 100]:
        print('Proportion', k, '% :')
        classifier = Classifier(k=k)
        classifier.adaboost_fit(cross_validation=False)
        classifier.adaboost_accuracy()
    print('=' * 55)
    for k in [5, 10, 20, 50, 100]:
        print('Proportion', k, '% :')
        classifier = Classifier(k=k)
        classifier.neural_net_fit(cross_validation=False)
        classifier.neural_net_accuracy()


def q_4():
    classifier.feature_selection()


def q_5():
    classifier.best_feature_method()


q_2()
