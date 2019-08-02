import pandas as pd
import numpy as np
from sklearn.utils import shuffle

class GQ_MS:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, file, normalize=True, logxfm = False, shuffledata=True):

        trn = load_data_and_clean_and_split(file, normalize, logxfm, shuffledata)

        self.trn = self.Data(trn)

        self.n_dims = self.trn.x.shape[1]


def load_data(file, normalize=True, logxfm = False, shuffledata=True):

    # data = pd.read_pickle(file)
    # data = pd.read_pickle(file).sample(frac=0.25)
    # data.to_pickle(file)
    if '.xlsx' in file:
        data = pd.read_excel(file)
    else:
        data = pd.read_csv(file)
    if shuffledata:
        data = shuffle(data)
        data.reset_index(inplace=True, drop=True)
    if logxfm:
        data.iloc[:,1] = np.log(data.iloc[:,1])
        data.iloc[:, 3] = np.log(data.iloc[:, 3])
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
    if normalize:
        return (data - data.mean()) / data.std()
    else:
        return data

#
# def get_correlation_numbers(data):
#     C = data.corr()
#     A = C > 0.98
#     B = A.values.sum(axis=1)
#     return B
#
#
# def load_data_and_clean(file):
#
#     data = load_data(file)
#     B = get_correlation_numbers(data)
#
#     while np.any(B > 1):
#         col_to_remove = np.where(B > 1)[0][0]
#         col_name = data.columns[col_to_remove]
#         data.drop(col_name, axis=1, inplace=True)
#         B = get_correlation_numbers(data)
#     # print(data.corr())
#     data = (data - data.mean()) / data.std()
#
#     return data


def load_data_and_clean_and_split(file, normalize=True, logxfm = False, shuffledata=True):

    return load_data(file, normalize, logxfm, shuffledata).values
