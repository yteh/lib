def checkDfInfo(df):
    '''
    - df: pandas DataFrame
    
    return a pretty print DataFrame info
    '''
    
    columns = df.columns
    nb_rows = df.shape[0]
    
    print('Number of rows   : ', nb_rows)
    print('Number of columns: ', len(columns))
    
    print('=' * 98)
    print('{:3}  | {:40} | {:15} | {:30}'.format('Idx', 'Name', 'dtype', 'Nb of Missing Values'))
    print('=' * 98)
    
    for i, col in enumerate(columns):
        nb_missing = df[col].isna().sum()
        pr_missing = nb_missing / nb_rows * 100
        print('{:3}) | {:40} | {:15} | {:20} ({:6.2f}%)'.format(i, col, str(df[col].dtype), nb_missing, pr_missing))
    
    print('=' * 98)