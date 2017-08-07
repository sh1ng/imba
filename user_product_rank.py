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

    order_train = pd.read_pickle(os.path.join(path, 'chunk_0.pkl'))

    orders_products = pd.merge(orders, order_prior, on="order_id")

    orders_products_products = pd.merge(orders_products, products[['product_id', 'department_id', 'aisle_id']],
                                        on='product_id')

    user_dep_stat = orders_products_products.groupby(['user_id', 'department_id']).agg(
        {'product_id': lambda x: x.nunique(),
         'reordered': 'sum'
         })
    print(user_dep_stat.columns)
    user_dep_stat.rename(columns={'product_id': 'dep_products',
                                  'reordered': 'dep_reordered'}, inplace=True)
    user_dep_stat.reset_index(inplace=True)
    print(user_dep_stat.columns)
    user_dep_stat.to_pickle('data/user_department_products.pkl')

    user_aisle_stat = orders_products_products.groupby(['user_id', 'aisle_id']).agg(
        {'product_id': lambda x: x.nunique(),
         'reordered': 'sum'
         })
    print(user_aisle_stat.columns)
    user_aisle_stat.rename(columns={'product_id': 'aisle_products',
                                    'reordered': 'aisle_reordered'}, inplace=True)
    user_aisle_stat.reset_index(inplace=True)
    user_aisle_stat.to_pickle('data/user_aisle_products.pkl')