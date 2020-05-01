import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns


def checkDfInfo(df, only_na=False):
    """
    - df: pandas DataFrame
    
    return a pretty print DataFrame info
    """
    
    col_w_na = [col for col in df.columns if df[col].isna().sum() > 0]
    columns  = df.columns if only_na == False else col_w_na
    nb_rows  = df.shape[0]
    nb_numerical_feat = len(df.select_dtypes(exclude=['object']).columns)
    nb_categorical_feat = len(df.select_dtypes(include=['object']).columns)
    
    print('Number of examples: ', nb_rows)
    print('Number of features: ', len(df.columns))
    print('Number of numerical features: ', nb_numerical_feat)
    print('Number of categorical features: ', nb_categorical_feat)
    print('Number of features with missing value: ', len(col_w_na))
    
    print('=' * 93)
    print('{:3}  | {:40} | {:15} | {:25}'.format('Idx', 'Name', 'dtype', 'Nb of Missing Values (%)'))
    print('=' * 93)
    
    for i, col in enumerate(columns):
        nb_missing = df[col].isna().sum()
        pr_missing = nb_missing / nb_rows * 100
        print('{:3}) | {:40} | {:15} | {:15} ({:6.2f}%)'.format(i, col, str(df[col].dtype), nb_missing, pr_missing))
    
    print('=' * 93)
    print()


def plotPearsonCorr(df, figsize=(10,10)):
    """
    - df: pandas DataFrame with both input and target variables. 

    Note: Please ensure all the value are numerical

    return a Pearson Correlation heatmap
    """

    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # mask the upper right triangle
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', cbar=False)
    ax.set_title('Pearson Correlation')
    plt.show()


def plotFeatureImportance(feature_importance, columns, n_largest=None, figsize=(12,12)):
    """
    - feature_importance: 1-D array/list contain the coef for each features
    - columns           : 1-D list of features name
    - n_largest         : top n importance features to plot (default = plot all)

    return a horizontal bar plot of n_largest feature importance
    """

    n_largest  = n_largest if n_largest is not None else len(columns)
    feat_imp   = pd.DataFrame(feature_importance, index=columns).reset_index()
    feat_imp.columns = ['Features', 'Coef']
    top_n_feat = feat_imp.nlargest(n_largest, 'Coef')

    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Coef', y='Features', data=top_n_feat)
    ax.set_title(f'Top %d Feature Importance' % n_largest)
    plt.show()
