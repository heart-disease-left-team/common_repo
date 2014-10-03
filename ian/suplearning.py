""" This module is meant to automate supervised learning tasks and tests.

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve


class Classifiers():
    """ Class to organize and simply classifcation algorithms.
        Will automatically split data into training, test sets.
        Passes through arguments to sklearn classifying algortithms.

        See available models with self.modelnames or
        self.models

        Initialize with Classifiers(Xdata, ydata, scale=True)

        Defaults
            -Scales data linearly. Set scale=False
        avoid. Unscaled data always found at self.Xdata_unscaled
            -Splits data to 75% training, 25% testing.
    """

    ###########################################################################
    # To add model, add in information to models, and import it above. ########
    models = {
                'dt': {'name': 'DecisionTreeClassifier',
                      'module': 'sklearn.tree.DecisionTreeClassifier'},
                'bnb': {'name': 'BernoulliNB',
                        'module': 'sklearn.naive_bayes.BernoulliNB'},
                'gnb': {'name': 'GaussianNB',
                        'module': 'sklearn.naive_bayes.GaussianNB'},
                'knn': {'name': 'KNeighborsClassifier',
                        'module': 'sklearn.neighbors.KNeighborsClassifier'},
                'log': {'name': 'LogisticRegression',
                        'module': 'sklearn.linear_model.LogisticRegression'},
                'rf': {'name': 'RandomForestClassifier',
                        'module': 'sklearn.ensemble.RandomForestClassifier'},
                'svc': {'name': 'SVC',
                        'module': 'sklearn.ensemble.SVC'}
                }
    ###########################################################################

    Xdata_orig = None
    Xdata = None
    ydata = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    modelnames = {}


    def __init__(self, Xdata, ydata, scale=True, pos_label=1):
        """ To not scale Xdata, set scale=False.
            To set y value that counts as a positive result for
            ROC and AUC, set pos_label=
        """
        self.Xdata_orig = Xdata

        if scale == True:
            self.Xdata = preprocessing.scale(self.Xdata_orig)
        elif scale == False:
            self.Xdata = self.Xdata_orig
        else:
            print 'Unusable value for scale='
            raise TypeError

        self.ydata = ydata
        self.pos_label= pos_label
        self.train_test_split()

        for model in self.models.keys():
            shortname = model
            longname = self.models[model]['name']
            t = {shortname: longname}
            self.modelnames.update(t)
            self.model = None


    def scoring(self, ytrue, ypredict, model=None):
        """Calulate accuracy score and classification report.
            If 'model=' is assigned, save these stats in
            self.models['model'] dictionary.
        """
        accuracy_score = metrics.accuracy_score(ytrue, ypredict)
        classification_report = metrics.classification_report(ytrue, ypredict)
        print 'Accuracy Score: %.4f' % accuracy_score
        print classification_report
        if model != None:
            assert model in self.modelnames
            self.models[model].update({'accuracy': accuracy_score,
                        'classification_report': classification_report})

    def learning_curve(self, model, *args, **kwargs):
        """ Plot learning curve for given model
            Note: learning_curve does cross-validation!
            Can use shortmodel names or call directly with arguments.

            Ex: learning_curve('gnb') OR
                learning_curve(GaussianNB(keyword=4))

            m, train_err and test_err lists saved in
            self.models[model][keyword].
        """
        # Note -- learning_curve does cross-validation
        assert model in self.modelnames
        modellong = self._get_model_longname(model)
        t = 'modelinstance = ' + modellong + '(*args, **kwargs)'
        exec t

        m, train_err, test_err = learning_curve(modelinstance, self.Xdata,
                                                self.ydata)
        train_cv_err = np.mean(train_err, axis=1)
        test_cv_err = np.mean(test_err, axis=1)

        self.models[model]['m'] = m
        self.models[model]['train_err'] = train_cv_err
        self.models[model]['test_err'] = test_cv_err

        titlestring = modellong + ' Learning Curve'
        plt.figure()
        plt.plot(m, train_cv_err, label='Training')
        plt.plot(m, test_cv_err, label='Testing')
        plt.title(titlestring)
        plt.legend()

    def roc_auc(self, model, *args, **kwargs):
        """ Graph ROC and calculate auc for model.
            Pass through to sklearn.metrics.rock_curve

            To select index of feature in Xdata that counts as
            positive, set featurenum=(feature number) when
            initializing class.

            Sets self.models[model] 'predict_proba', 'fpr',
            'tpr', 'threshold', 'auc'.
        """
        assert model in self.models.keys()
        t = 'score = self.' + model + '.predict_proba(self.X_test)'
        exec t
        # Pick out feature we want to count as positive
        exec 'classes = self.' + model + '.classes_'
        featurenum = list(classes).index(self.pos_label)
        scoreA = [row[featurenum] for row in score]
        p = int(self.pos_label)
        ## Some weird bug with pos_label actually being passed
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test.astype(int),
                scoreA, pos_label=p)
        auc = metrics.roc_auc_score(self.y_test, scoreA)
        print 'AUC = %.3f' % auc

        self.models[model]['predict_proba'] = score
        self.models[model]['fpr'] = fpr
        self.models[model]['tpr'] = tpr
        self.models[model]['threshold'] = thresholds
        self.models[model]['auc'] = auc

        # plt.figure()
        # plt.plot(fpr, tpr, label=model)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend(loc=4)


    def train_test_split(self, test_size=0.25, **kwargs):
        """Pass through to sklearn.cross_validation_train_test_split()
           Default is 75% training, 25% testing. Change with 'test_size='.
        """
        if (self.X_train != None) or (self.y_train != None):
            print 'WARNING! Overwriting training and testing sets!'
        self.X_train, self.X_test,  self.y_train, self.y_test = train_test_split(
                                self.Xdata, self.ydata, test_size=test_size, **kwargs)

    def knn_findmaxk(self, kmax=30, **kwargs):
        """ Try different k values from one to kmax=30.
            Sets self.models['knn']['k_bestfit'] as k value with highest accuracy.
            Pass through to sklearn.neighbors.KNeighborsClassifier()
        """
        kmaxscore = 0
        kmaxvalue = None
        for k in range(1, kmax + 1):
            knn = KNeighborsClassifier(n_neighbors=k, **kwargs)
            knn.fit(self.X_train, self.y_train)
            y_predict = knn.predict(self.X_test)
            score = metrics.accuracy_score(self.y_test, y_predict)
            if score > kmaxscore:
                kmaxscore = score
                kmaxvalue = k
        self.models['knn']['k_bestfit'] = kmaxvalue

    def do_modeling_all(self, learncurve=False):
        """ Run self.do_modeling on all models
        """
        for model in self.modelnames.keys():
            print '========================================================='
            print self._get_model_longname(model)
            if model == 'svc':
                self.do_modeling(model, learncurve=learncurve, probability=True)
            else:
                self.do_modeling(model, learncurve=learncurve)

        try:
            plt.figure()
            for model in self.modelnames.keys():
                fpr = self.models[model]['fpr']
                tpr = self.models[model]['tpr']
                plt.plot(fpr, tpr, label=model)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc=4)
        except KeyError:
            print """No ROC plots due to 'fpr' or 'tpr' KeyError"""


    def do_modeling(self, model, learncurve=True,
                    rocauc=True, *args, **kwargs):
        """ Select model, do model.fit, model.predict and calculate scores.
            Takes one argument -- shortname for model. List of valid shortnames
            found in self.modelnames

            To calculate learning curves, set learncurve=True

            Prototype: do_modeling('knn')

            Model saved as self.model (ie self.knn).
            Predicted values, scores, etc. saved in self.models[model]
                i.e. self.models['knn'].
        """
        assert model in self.models.keys()
        modellong = self._get_model_longname(model)

        # Create commands
        model_str = """'""" + model + """'"""
        selfmodel = 'self.' + model
        selfmodeldict = 'self.models[' + model_str + ']'

        init_com = selfmodel + ' = ' + self.modelnames[model] + '(*args, **kwargs)'
        fit_com = selfmodel + '.fit(self.X_train, self.y_train)'
        predict_com = 'y_predict = ' + selfmodel + '.predict(self.X_test)'
        save_predict_com = selfmodeldict + """['y_predict'] = y_predict"""

        scoring_com1 = 'self.scoring(self.y_test, '+ selfmodeldict
        scoring_com2 = """['y_predict'], model=""" + model_str + ')'
        scoring_com = scoring_com1 + scoring_com2

        learning_curve_com1 = 'self.learning_curve(' + model_str
        learning_curve_com2 = ', *args, **kwargs)'
        learning_curve_com = learning_curve_com1 + learning_curve_com2

        roc_auc_com = 'self.roc_auc(' + model_str +')'

        # Execute commands
        exec init_com
        exec fit_com
        exec predict_com
        exec save_predict_com
        exec scoring_com
        if learncurve:
            exec learning_curve_com
        if rocauc:
            exec roc_auc_com

    def _get_model_longname(self, model):
        """ Used to convert shorthand name for classifiers to offical
            sklearn name. Preferred method.
        """
        return self.modelnames[model]


