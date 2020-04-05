def checkDfInfo(df):
    """
    - df: pandas DataFrame
    
    return a pretty print DataFrame info
    """
    
    columns = df.columns
    nb_rows = df.shape[0]
    
    print('Number of rows   : ', nb_rows)
    print('Number of columns: ', len(columns))
    
    print('=' * 93)
    print('{:3}  | {:40} | {:15} | {:25}'.format('Idx', 'Name', 'dtype', 'Nb of Missing Values (%)'))
    print('=' * 93)
    
    for i, col in enumerate(columns):
        nb_missing = df[col].isna().sum()
        pr_missing = nb_missing / nb_rows * 100
        print('{:3}) | {:40} | {:15} | {:15} ({:6.2f}%)'.format(i, col, str(df[col].dtype), nb_missing, pr_missing))
    
    print('=' * 93)


def plotFeatureImportance(feature_importance, columns, n_largest=None):
    """
    - feature_importance: 1-D array/list contain the coef for each features
    - columns           : list of features name
    - n_largest         : top n importance features to plot (default = plot all)

    return a horizontal bar plot of n_largest feature importance
    """

    n_largest  = n_largest if n_largest is not None else len(columns)
    feat_imp   = pd.DataFrame(feature_importance, index=columns).reset_index()
    feat_imp.columns = ['Features', 'Coef']
    top_n_feat = feat_imp.nlargest(n_largest, 'Coef')

    ax = sns.barplot(x='Coef', y='Features', data=top_n_feat)
    ax.set_title(f'Top %d Feature Importance' % n_largest)
    plt.show()
