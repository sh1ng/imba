import gc
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    path = "data"

    order_prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"), dtype={'order_id': np.uint32,
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

    print('loaded')

    orders = orders.loc[orders.eval_set == 'prior', :]
    orders_user = orders[['order_id', 'user_id']]
    labels = pd.merge(order_prior, orders_user, on='order_id')
    labels = labels.loc[:, ['user_id', 'product_id']].drop_duplicates()

    print(labels)

    print('save')
    print(labels.shape)
    print(labels.columns)
    labels.to_pickle('data/previous_products.pkl')