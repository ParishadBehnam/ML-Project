import pandas as pd
import numpy as np


def prepare_test_data():
    # take 10% of data as test set
    df_x = pd.read_csv('../data/Train/X_train.txt', sep=" ", header=None)
    df_y = pd.read_csv('../data/Train/y_train.txt', sep=" ", header=None)
    df_si = pd.read_csv('../data/Train/subject_id_train.txt', sep=" ", header=None)
    indices = np.random.choice(df_y.shape[0], int(0.1*df_y.shape[0]))

    np.savetxt('../data/Test/X_test.txt', df_x.take(indices, axis=0).values, fmt='%f')
    np.savetxt('../data/Test/y_test.txt', df_y.take(indices, axis=0).values, fmt='%d')
    np.savetxt('../data/Test/subject_id_test.txt', df_si.take(indices, axis=0).values, fmt='%d')

    np.savetxt('../data/Train/X_train.txt', df_x.drop(labels=indices, axis=0).values, fmt='%f')
    np.savetxt('../data/Train/y_train.txt', df_y.drop(labels=indices, axis=0).values, fmt='%d')
    np.savetxt('../data/Train/subject_id_train.txt', df_si.drop(labels=indices, axis=0).values, fmt='%d')


prepare_test_data()
