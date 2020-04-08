import pandas as pd

from sklearn.preprocessing import LabelEncoder


class DataTransform:
    """
    Attributes:
    - label_encoder_ : dict containing a sklearn LabelEncoder for each encoded columns

    """

    def __init__(self):
        self.label_encoder_ = {}


    def labelEncoder(self, df, cols=[]):
        """
        - df   : pandas DataFrame
        - cols : a list of columns name to encode

        return encoded DataFrame
        """

        for col in cols:
            # transform each label into numerical value
            le      = LabelEncoder()
            df[col] = le.fit_transform(df[col])

            # save each encoder for future reverse transform
            self.label_encoder_[col] = le

        return df


    @staticmethod
    def oneHotEncoded(df, cols=[]):
        """
        - df   : pandas DataFrame
        - cols : a list of columns name to encode

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


    