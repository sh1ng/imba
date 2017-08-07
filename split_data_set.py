import gc
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    path = "data"
    folds = 1

    aisles = pd.read_csv(os.path.join(path, "aisles.csv"), dtype={'aisle_id': np.uint8, 'aisle':'category'})
    departments = pd.read_csv(os.path.join(path, "departments.csv"), dtype={'department_id':np.uint8, 'department': 'category'})
    order_prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order':np.uint8,
                                                                                      'reordered': bool})
    order_train = pd.read_csv(os.path.join(path, "order_products__train.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order':np.uint8,
                                                                                      'reordered': bool})
    orders = pd.read_csv(os.path.join(path, "orders.csv"), dtype={'order_id':np.uint32,
                                                                  'user_id': np.uint32,
                                                                  'eval_set': 'category',
                                                                  'order_number':np.uint8,
                                                                  'order_dow': np.uint8,
                                                                  'order_hour_of_day': np.uint8
                                                                  })

    products = pd.read_csv(os.path.join(path, "products.csv"), dtype={'product_id': np.uint16,
                                                                      'aisle_id': np.uint8,
                                                                      'department_id': np.uint8})

    labels = pd.read_pickle('data/previous_products.pkl')
    orders = orders.loc[(orders.eval_set == 'train') | (orders.eval_set == 'test'), :]
    labels = pd.merge(labels, orders[['order_id', 'user_id', 'eval_set']], on='user_id').drop(['user_id'], axis=1)

    order_train.drop(['add_to_cart_order'], axis=1, inplace=True)

    print('data is loaded')

    orders = np.unique(labels.order_id)

    size = orders.shape[0] // folds

    for fold in range(folds):

        current = orders[fold * size:(fold + 1) * size]

        current = labels.loc[np.in1d(labels.order_id, current), :]

        current = pd.merge(order_train, current, on=['order_id', 'product_id'], how='right')
        current.reordered.fillna(False, inplace=True)
        print(current.columns)
        print(current.shape)

        current.to_pickle('data/chunk_{}.pkl'.format(fold))
