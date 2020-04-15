"""
TODO:
1. create a function which can help auto generate schema file

"""

import pandas as pd
import numpy  as np
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.impute        import SimpleImputer


class DataTransform:
    """
    Attributes:
    - label_encoder_ : dict containing a sklearn LabelEncoder for each encoded columns

    """

    def __init__(self):
        self.label_encoder_ = {}
        self.imputer_ = {}


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
            df.drop(columns=[col], axis=1)

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
