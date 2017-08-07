import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

from utils import fast_search

none_product = 50000

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def create_products(df):
    # print(df.product_id.values.shape)
    products = df.product_id.values
    prob = df.prediction.values

    sort_index = np.argsort(prob)[::-1]

    values = fast_search(prob[sort_index][0:80], dtype=np.float64)

    index = np.argmax(values)

    print('iteration', df.shape[0], 'optimal value', index)

    best = ' '.join(map(lambda x: str(x) if x != none_product else 'None', products[sort_index][0:index]))
    df = df[0:1]
    df.loc[:, 'products'] = best
    return df

if __name__ == '__main__':
    data = pd.read_pickle('data/prediction_rnn.pkl')
    data['not_a_product'] = 1. - data.prediction

    gp = data.groupby('order_id')['not_a_product'].apply(lambda x: np.multiply.reduce(x.values)).reset_index()
    gp.rename(columns={'not_a_product': 'prediction'}, inplace=True)
    gp['product_id'] = none_product

    data = pd.concat([data, gp], axis=0)
    data.product_id = data.product_id.astype(np.uint32)

    data = data.loc[data.prediction > 0.01, ['order_id', 'prediction', 'product_id']]

    data = applyParallel(data.groupby(data.order_id), create_products).reset_index()

    data[['order_id', 'products']].to_csv('data/sub.csv', index=False)
