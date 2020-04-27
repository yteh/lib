import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

"""
TODO:
    1. Fixed the bug when the class label does not start at 1
"""

class Multiclassification:

    @staticmethod
    def perClassAccuracy(y_true, y_pred):
        """
        - y_true: 1-D list/array of actual labels
        - y_pred: 1-D list/array of predicted labels

        return scalar value of the average class accuracy
        """

        # ensure y_true and y_pred is in numpy format
        if type(y_true).__module__ != np.__name__: y_true = np.array(y_true)
        if type(y_pred).__module__ != np.__name__: y_pred = np.array(y_pred)

        # get all the unique labels
        unq_labels = list(set(y_true))

        # scores with take multiclass and imbalance
        # problem into consideration
        cks = cohen_kappa_score(y_true, y_pred, labels=unq_labels)

        # one hot encoding
        ytrue_ohe = np.zeros((y_true.shape[0], y_true.max()+1))
        ypred_ohe = np.zeros((y_pred.shape[0], y_pred.max()+1))
        ytrue_ohe[np.arange(y_true.shape[0]), y_true] = 1
        ypred_ohe[np.arange(y_pred.shape[0]), y_pred] = 1


        # calculate the accuracy per class
        accuracy = []

        for label in unq_labels:
            cm    = confusion_matrix(ytrue_ohe[:, label], ypred_ohe[:, label])
            count = cm[(0,1), (0,1)].sum()         # count the true positive and negative
            score = count.sum() / cm.sum() * 100   # calculate the accuracy by diving with the total number of samples
            accuracy.append(score)

        # show result
        print('Cohen Kappa score: {:.3f}'.format(cks))

        ax = sns.heatmap([accuracy], square=True, cbar=False, annot=True, fmt='.2f')
        ax.set_title('per Class Accuracy')
        ax.set_ylabel('Acc (%)')
        ax.set_xlabel('Class')
        ax.set_yticklabels([])
        ax.set_xticklabels(unq_labels)
        plt.show()

        return cks



