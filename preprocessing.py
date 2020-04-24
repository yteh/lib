"""
TODO:
1. create a function which can help auto generate schema file

"""

import pandas as pd
import numpy  as np
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.impute        import SimpleImputer
from scipy.stats           import skew, boxcox


class DataTransform:
    """
    Attributes:
    - label_encoder_ : dict containing a sklearn LabelEncoder for each encoded columns

    """

    def __init__(self):
        self.label_encoder_ = {}
        self.imputer_ = {}
        self.skew_transform_ = {}


    def labelEncoder(self, df, cols=[]):
        """
        - df   : pandas DataFrame
        - cols : 1-D list of columns name to encode

        return encoded DataFrame
        """

        for col in cols:
            # transform each label into numerical value
            le      = LabelEncoder()
            df[col] = le.fit_transform(df[col])

            # save each encoder for future reverse transform
            self.label_encoder_[col] = le

        return df


    def labelEncoderTransform(self, df):
        """
         - df   : pandas DataFrame

        return encoded DataFrame
        """

        if not self.label_encoder_: 
            print('Please call labelEncoder Method before calling transform')
            return df

        for col in self.label_encoder_:
            df[col] = self.label_encoder_[col].transform(df[col])
            
        return df

    @staticmethod
    def oneHotEncoded(df, cols=[]):
        """
        - df   : pandas DataFrame
        - cols : 1-D list of columns name to encode

        return one hot encoded DataFrame
        """

        for col in cols:
            # one hot encode each column and 
            # merge it with the original df
            dummies = pd.get_dummies(df[col], prefix=col)
            df      = pd.concat([df, dummies], axis=1)

        # drop the orginal column
        df = df.drop(columns=cols, axis=1)

        return df


    def missingValueImputer(self, df, cols=[], strategy='mean'):
        """
        - df       : pandas DataFrame
        - cols     : 1-D list of columns name to encode
        - strategy : string ('mean','median,'most_frequent'), the statistical method used to imputer the missing value

        return imputed dataframe
        """

        for col in cols:
            imputer = SimpleImputer(strategy=strategy)
            df[col] = imputer.fit_transform(df[col])

            self.imputer_[col] = imputer

        return df


    def missingValueImputerTransform(self, df):
        """
        - df     : pandas DataFrame

        return transformed DataFrame
        """

        for col in self.imputer_:
            df[col] = self.imputer_[col].transform(df[col])

        return df


    def skewTransform(self, df, verbose=True):
        """
        - df  : pandas DataFrame

        return DataFrame with log transformed on outlier features
        """

        # fail safe, to prevent error
        # find the numerical features
        numerical_features = df.select_dtypes(exclude=['object']).columns
        numeric_df = df[numerical_features]

        # calculate the skewness for each features
        skewness = numeric_df.apply(lambda x: skew(x))
        skewness = skewness[abs(skewness) > 0.5]

        # record the default skewness (only if the skewness > 0.5)
        self.skew_transform_ = {feature: {'before': val, 'after': val, 'method': 'default'} \
                                for feature, val in skewness.to_dict().items()}

        # obtain the features name for the features that has skewness > 0.5
        skew_features = skewness.index

        # log transfrom the skew features
        transformed = np.array([self.__autoTransform(df, col) for col in skew_features]).T

        # print the number of features transform
        if verbose: 
            print(' %d features have skewness > 0.5' % skew_features.shape[0])

            for idx, col in enumerate(skew_features):
                if self.skew_transform_[col]['method'] == 'default': continue

                fig, axs = plt.subplots(1, 2)
                ax = sns.distplot(df[col], ax=axs[0], bins=100)
                ax.set_title(f'%s (Before, Skewness: %.3f)' % (col, self.skew_transform_[col]['before']))
                
                ax = sns.distplot(transformed[:, idx], ax=axs[1], bins=100)
                ax.set_title(f'%s (After, Method: %s, Skewness: %.3f)' % (col, self.skew_transform_[col]['method'], self.skew_transform_[col]['after']))
                plt.show()

        print(transformed.shape)
        print(df[skew_features].shape)
        df[skew_features] = transformed

        return df


    def __autoTransform(self, df, col):
        methods      = ['log1p', 'sqrt', 'boxcox']

        # boxcox can work on positive value only
        if df[col].min() > 0:
            transformed  = [np.log1p(df[col]), np.sqrt(df[col]), boxcox(df[col])[0]]

        # exclude boxcox transform if negative value present
        else:
            transformed  = [np.log1p(df[col]), np.sqrt(df[col])]

        # calculate the skewness on each column
        skewness     = list(map(lambda x: skew(x), transformed))
        max_skew_idx = np.argmax(skewness)

        # check and record the skewness for each column
        if skewness[max_skew_idx] < self.skew_transform_[col]['after']:
            self.skew_transform_[col]['after']  = skewness[max_skew_idx]
            self.skew_transform_[col]['method'] = methods[max_skew_idx]

            return transformed[max_skew_idx]

        return df[col]


