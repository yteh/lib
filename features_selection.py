import statsmodels.api   as sm
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns

from sklearn.feature_selection import RFE
from sklearn.model_selection   import train_test_split, GridSearchCV
from sklearn.metrics           import mean_squared_error as mse
from sklearn.base              import clone
from sklearn.linear_model      import LassoCV

def plotScores(scores, metric_name='RMSE'):
    """
    scores : list of score per iteration

    return a line plot of scores per iteration
    """

    plt.plot(range(len(scores)), scores, marker= "o")
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.title(f'%s Score per Iteration' % metric_name)
    plt.show()


class RegressionSelector:
    """
    this call only work on data with regression problem.
    specifically features with continuos value 
    """

    def __init__(self):
        self.elim_feat_ = {}

    
    def __recordEliminatedFeature(self, col):
        if col not in self.elim_feat_:
            self.elim_feat_[col] = 1
        else: 
            self.elim_feat_[col] += 1


    def plotEliminatedFeatures(self):
        """
        return a barplot of eliminated features by the Elimination method called
        """

        if not self.elim_feat_:
            print('Please called \'Elimination\' method before plotting')
            return False


        cols, rank = self.elim_feat_.keys(), self.elim_feat_.values()

        ax = sns.barplot(x=cols, y=rank)
        ax.set_title('Ranking for each Eliminated Features')
        ax.set_xlabel('Eliminated Features')
        ax.set_ylabel('Rank')
        plt.show()


    def backwardElimination(self, df, target, p_thresh=0.05, verbose=True):
        """
        - df       : pandas DataFrame
        - target   : string, the name of the target
        - p_thresh : float, the pvalue
        - verbose  : print the features removal process and plot model score (default = True)

        return list of selected features
        """
        rmse = []
        cols = list(df.drop(columns=[target], axis=1).columns)

        for i in range(len(cols)):
            # define the input and target variables
            x = df[cols]
            x = sm.add_constant(x)
            y = df[target]

            # fit the paired input to Oridinary Least Square model
            # to obtain the pvalue for each features
            model = sm.OLS(y, x).fit()
            pvalues = pd.Series(model.pvalues.values[1:], index=cols)

            # check if the pvalue is higher than p_thresh
            max_pvalue    = pvalues.max()
            idxmax_pvalue = pvalues.idxmax()

            # remove the feature with highest pvalue
            if max_pvalue > p_thresh:
                rmse_score = np.sqrt(mse(y, model.predict(x)))
                if verbose: print('Iter {:3}, Removing {:20} (pvalue = {:.2f}, RMSE = {:.3f})'.format(i+1, idxmax_pvalue, max_pvalue, rmse_score))

                cols.remove(idxmax_pvalue)
                rmse.append(rmse_score)
                self.__recordEliminatedFeature(cols[idxmax_pvalue])
            else:
                break

        # plot 
        if verbose: 
            print()
            plotScores(rmse)

        return cols


    def recursiveFeatureElimination(self, df, target, model, verbose=True):
        """
        - df      : pandas DataFrame, containing both input and target
        - target  : string, the name of the target
        - model   : sklearn regression model
        - verbose : print the model performance (default = True)

        return list of selected features
        """
        
        x      = df.drop(columns=[target], axis=1)
        y      = df[target]
        scores = []

        # split the data into train and test
        # to evaluate the model performance
        # on unseen data (test)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.1, random_state=42)

        # find the optimal number of features
        # for the regression model
        for n in range(1, len(x.columns)+1):
            # create a copy of the model
            model_cp = clone(model)
            rfe      = RFE(model_cp, n)
            
            # transform the dataset to have n features
            x_train_rfe = rfe.fit_transform(x_train, y_train)
            x_test_rfe  = rfe.transform(x_test)

            # check the r2 score on the dataset with n features
            model_cp.fit(x_train_rfe, y_train)
            scores.append(model_cp.score(x_test_rfe, y_test))

        # nof = number of features
        optimal_nof = scores.index(max(scores))

        # use RFE to select the best features
        rfe = RFE(model, optimal_nof)
        x_rfe = rfe.fit_transform(x, y)

        model.fit(x_rfe, y)
        feat_imp = pd.Series(rfe.support_, index=x.columns)
        feat_drp = feat_imp[feat_imp == False].index
        feat_imp = feat_imp[feat_imp == True].index

        # record the eliminated features
        list(map(self.__recordEliminatedFeature, feat_drp))

        if verbose: 
            print('Optimal Number of Features: {} (R2 score = {:.3f})'.format(optimal_nof, max(scores)))
            print('Drop Features: ', feat_drp)
            print('Selected Features: ', feat_imp)
            print()
            plotScores(scores, metric_name='R2 Score')

        return feat_imp


    def LassoElimination(self, df, target, verbose=True):
        """
        - df      : pandas DataFrame, containing both input and target
        - target  : string, the name of the target
        - verbose : print the feature importance (default = True)

        return list of selected features
        """

        x = df.drop(columns=[target])
        y = df[target]

        alphas = [6, 3, 1, 0.6, 0.3, 0.06, 0.03]
        model  = LassoCV(alphas=alphas)
        model.fit(x, y)
        feat_imp = pd.Series(model.coef_, index=x.columns)
        feat_drp = feat_imp[feat_imp <= 0].index
        feat_imp = feat_imp[feat_imp > 0].index

        # record the eliminated features
        list(map(self.__recordEliminatedFeature, feat_drp))

        if verbose:
            print('Optimal alpha value: {} (R2 score = {:.3f})\n'.format(model.alpha_, model.score(x, y)))

            ax = sns.heatmap(np.sqrt(model.mse_path_), annot=True, fmt='.2f', cmap='RdBu_r')
            ax.set_xlabel('N Fold')
            ax.set_ylabel('N Alphas')
            ax.set_yticklabels(alphas)
            ax.set_title('MSE score per alpha/folds')
            plt.show()

        return feat_imp[feat_imp > 0].index


    @staticmethod
    def singleValueElimination(df, cols=[], verbose=True):
        """
        - df      : pandas DataFrame, containing both input and target
        - cols    : 1-D list of available columns for elimination
        - verbose : print the feature importance (default = True)

        return list of selected features
        """

        cols = list(df.columns) if len(cols) == 0 else list(cols)
        singleValueFilter = lambda num: True if num == 1 else False

        # loop through each feature
        for idx, col in enumerate(cols):

            # count the unq value on each feature
            unq_count  = df[col].value_counts().to_dict()
            key, val   = unq_count.keys(), unq_count.values()
            nb_single = sum(list(map(singleValueFilter, val)))  # calculate the number of features with single value

            # check if all the unique has only one unique value
            if nb_single == len(key) or nb_single == len(key) - 1:
                del cols[idx]

                if verbose: print('Dropping feature {:3}: {}'.format(idx, col))

        print()

        return cols
            











