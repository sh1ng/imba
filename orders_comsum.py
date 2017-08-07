import gc
import pandas as pd
import numpy as np
import os
import json
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import dok_matrix, coo_matrix
from sklearn.utils.multiclass import  type_of_target


if __name__ == '__main__':
    path = "data"

    aisles = pd.read_csv(os.path.join(path, "aisles.csv"), dtype={'aisle_id': np.uint8, 'aisle': 'category'})
    departments = pd.read_csv(os.path.join(path, "departments.csv"),
                              dtype={'department_id': np.uint8, 'department': 'category'})
    order_prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order': np.uint8,
                                                                                      'reordered': bool})
    order_train = pd.read_csv(os.path.join(path, "order_products__train.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order': np.uint8,
                                                                                      'reordered': bool})
    orders = pd.read_csv(os.path.join(path, "orders.csv"), dtype={'order_id': np.uint32,
                                                                  'user_id': np.uint32,
                                                                  'eval_set': 'category',
                                                                  'order_number': np.uint8,
                                                                  'order_dow': np.uint8,
                                                                  'order_hour_of_day': np.uint8
                                                                  })

    products = pd.read_csv(os.path.join(path, "products.csv"), dtype={'product_id': np.uint16,
                                                                      'aisle_id': np.uint8,
                                                                      'department_id': np.uint8})

    labels = pd.read_pickle(os.path.join(path, 'chunk_0.pkl'))
    user_product = pd.read_pickle(os.path.join(path, 'previous_products.pkl'))

    order_comsum = orders[['user_id', 'order_number', 'days_since_prior_order']].groupby(['user_id', 'order_number'])\
            ['days_since_prior_order'].sum().groupby(level=[0]).cumsum().reset_index().rename(columns={'days_since_prior_order':'days_since_prior_order_comsum'})

    # order_comsum['days_since_prior_order_comsum'].fillna(0, inplace=True)
    order_comsum.to_pickle('data/orders_comsum.pkl')

    order_comsum = pd.merge(order_comsum, orders, on=['user_id', 'order_number'])[['user_id', 'order_number', 'days_since_prior_order_comsum', 'order_id']]

    order_product = pd.merge(order_prior, orders, on='order_id')[['order_id', 'product_id', 'eval_set']]
    order_product_train_test = labels[['order_id', 'product_id', 'eval_set']]

    order_product = pd.concat([order_product, order_product_train_test])

    order_product = pd.merge(order_product, order_comsum, on='order_id')

    print(order_product.columns)

    order_product = pd.merge(order_product, user_product, on=['user_id', 'product_id'])

    temp = order_product.groupby(['user_id', 'product_id', 'order_number'])['days_since_prior_order_comsum'].sum().groupby(level=[0, 1]).apply(lambda x: np.diff(np.nan_to_num(x)))
    temp = temp.to_frame('periods').reset_index()

    temp.to_pickle('data/product_period.pkl')

    aggregated = temp.copy()
    aggregated['last'] = aggregated.periods.apply(lambda x: x[-1])
    aggregated['prev1'] = aggregated.periods.apply(lambda x: x[-2] if len(x) > 1 else np.nan)
    aggregated['prev2'] = aggregated.periods.apply(lambda x: x[-3] if len(x) > 2 else np.nan)
    aggregated['median'] = aggregated.periods.apply(lambda x: np.median(x[:-1]))
    aggregated['mean'] = aggregated.periods.apply(lambda x: np.mean(x[:-1]))
    aggregated.drop('periods', axis=1, inplace=True)

    aggregated.to_pickle('data/product_periods_stat.pkl')