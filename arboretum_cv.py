import gc
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import os
import arboretum
import json
import sklearn.metrics
from sklearn.metrics import f1_score, roc_auc_score
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

    product_embeddings = pd.read_pickle('data/product_embeddings.pkl')
    embedings = list(range(32))
    product_embeddings = product_embeddings[embedings + ['product_id']]

    order_prev = pd.merge(order_train, orders, on='order_id')
    order_prev.order_number -= 1
    order_prev = pd.merge(order_prev[
                              ['user_id', 'order_number', 'product_id', 'reordered', 'add_to_cart_order', 'order_dow',
                               'order_hour_of_day']], orders[['user_id', 'order_number', 'order_id']],
                          on=['user_id', 'order_number'])

    order_prev.drop(['order_number', 'user_id'], axis=1, inplace=True)

    order_prev.rename(columns={
        'reordered': 'reordered_prev',
        'add_to_cart_order': 'add_to_cart_order_prev',
        'order_dow': 'order_dow_prev',
        'order_hour_of_day': 'order_hour_of_day_prev'
    }, inplace=True)

    products = pd.read_csv(os.path.join(path, "products.csv"), dtype={'product_id': np.uint16,
                                                                      'aisle_id': np.uint8,
                                                                      'department_id': np.uint8})

    order_train = pd.read_pickle(os.path.join(path, 'chunk_0.pkl'))
    order_train = order_train.loc[order_train.eval_set == "train", ['order_id',  'product_id',  'reordered']]

    product_periods = pd.read_pickle(os.path.join(path, 'product_periods_stat.pkl')).fillna(9999)
    # product_periods.prev1 = product_periods['last'] / product_periods.prev1
    # product_periods.prev2 = product_periods['last'] / product_periods.prev2
    # product_periods['mean'] = product_periods['last'] / product_periods['mean']
    # product_periods['median'] = product_periods['last'] / product_periods['median']

    print(order_train.columns)

    ###########################

    weights = order_train.groupby('order_id')['reordered'].sum().to_frame('weights')
    weights.reset_index(inplace=True)


    prob = pd.merge(order_prior, orders, on='order_id')
    print(prob.columns)
    prob = prob.groupby(['product_id', 'user_id'])\
        .agg({'reordered':'sum', 'user_id': 'size'})
    print(prob.columns)

    prob.rename(columns={'sum': 'reordered',
                         'user_id': 'total'}, inplace=True)

    prob.reordered = (prob.reordered > 0).astype(np.float32)
    prob.total = (prob.total > 0).astype(np.float32)
    prob['reorder_prob'] = prob.reordered / prob.total
    prob = prob.groupby('product_id').agg({'reorder_prob': 'mean'}).rename(columns={'mean': 'reorder_prob'})\
        .reset_index()

    prod_stat = order_prior.groupby('product_id').agg({'reordered': ['sum', 'size'],
                                                       'add_to_cart_order':'mean'})
    prod_stat.columns = prod_stat.columns.levels[1]
    prod_stat.rename(columns={'sum':'prod_reorders',
                              'size':'prod_orders',
                              'mean': 'prod_add_to_card_mean'}, inplace=True)
    prod_stat.reset_index(inplace=True)

    prod_stat['reorder_ration'] = prod_stat['prod_reorders'] / prod_stat['prod_orders']

    prod_stat = pd.merge(prod_stat, prob, on='product_id')

    # prod_stat.drop(['prod_reorders'], axis=1, inplace=True)

    user_stat = orders.loc[orders.eval_set == 'prior', :].groupby('user_id').agg({'order_number': 'max',
                                                                                  'days_since_prior_order': ['sum',
                                                                                                             'mean',
                                                                                                             'median']})
    user_stat.columns = user_stat.columns.droplevel(0)
    user_stat.rename(columns={'max': 'user_orders',
                              'sum': 'user_order_starts_at',
                              'mean': 'user_mean_days_since_prior',
                              'median': 'user_median_days_since_prior'}, inplace=True)
    user_stat.reset_index(inplace=True)

    orders_products = pd.merge(orders, order_prior, on="order_id")

    user_order_stat = orders_products.groupby('user_id').agg({'user_id': 'size',
                                                              'reordered': 'sum',
                                                              "product_id": lambda x: x.nunique()})

    user_order_stat.rename(columns={'user_id': 'user_total_products',
                                    'product_id': 'user_distinct_products',
                                    'reordered': 'user_reorder_ratio'}, inplace=True)
    user_order_stat.reset_index(inplace=True)
    user_order_stat.user_reorder_ratio = user_order_stat.user_reorder_ratio / user_order_stat.user_total_products

    user_stat = pd.merge(user_stat, user_order_stat, on='user_id')
    user_stat['user_average_basket'] = user_stat.user_total_products / user_stat.user_orders

    ########################### products

    prod_usr = orders_products.groupby(['product_id']).agg({'user_id': lambda x: x.nunique()})
    prod_usr.rename(columns={'user_id':'prod_users_unq'}, inplace=True)
    prod_usr.reset_index(inplace=True)

    prod_usr_reordered = orders_products.loc[orders_products.reordered, :].groupby(['product_id']).agg({'user_id': lambda x: x.nunique()})
    prod_usr_reordered.rename(columns={'user_id': 'prod_users_unq_reordered'}, inplace=True)
    prod_usr_reordered.reset_index(inplace=True)

    order_stat = orders_products.groupby('order_id').agg({'order_id': 'size'})\
        .rename(columns = {'order_id': 'order_size'}).reset_index()

    orders_products = pd.merge(orders_products, order_stat, on='order_id')
    orders_products['add_to_cart_order_inverted'] = orders_products.order_size - orders_products.add_to_cart_order
    orders_products['add_to_cart_order_relative'] = orders_products.add_to_cart_order / orders_products.order_size

    data_dow = orders_products.groupby(['user_id', 'product_id', 'order_dow']).agg({
                                                                   'reordered': ['sum', 'size']})
    data_dow.columns = data_dow.columns.droplevel(0)
    data_dow.columns = ['reordered_dow', 'reordered_dow_size']
    data_dow['reordered_dow_ration'] = data_dow.reordered_dow / data_dow.reordered_dow_size
    data_dow.reset_index(inplace=True)

    data = orders_products.groupby(['user_id', 'product_id']).agg({'user_id': 'size',
                                                                   'order_number': ['min', 'max'],
                                                                   'add_to_cart_order': ['mean', 'median'],
                                                                   'days_since_prior_order': ['mean', 'median'],
                                                                   'order_dow': ['mean', 'median'],
                                                                   'order_hour_of_day': ['mean', 'median'],
                                                                   'add_to_cart_order_inverted': ['mean', 'median'],
                                                                   'add_to_cart_order_relative': ['mean', 'median'],
                                                                   'reordered':['sum']})

    data.columns = data.columns.droplevel(0)
    data.columns = ['up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position', 'up_median_cart_position',
                             'days_since_prior_order_mean', 'days_since_prior_order_median', 'order_dow_mean', 'order_dow_median',
                             'order_hour_of_day_mean', 'order_hour_of_day_median',
                    'add_to_cart_order_inverted_mean', 'add_to_cart_order_inverted_median',
                    'add_to_cart_order_relative_mean', 'add_to_cart_order_relative_median',
                    'reordered_sum'
                    ]

    data['user_product_reordered_ratio'] = (data.reordered_sum + 1.0) / data.up_orders

    # data['first_order'] = data['up_orders'] > 0
    # data['second_order'] = data['up_orders'] > 1
    #
    # data.groupby('product_id')['']

    data.reset_index(inplace=True)

    data = pd.merge(data, prod_stat, on='product_id')
    data = pd.merge(data, user_stat, on='user_id')

    data['up_order_rate'] = data.up_orders / data.user_orders
    data['up_orders_since_last_order'] = data.user_orders - data.up_last_order
    data['up_order_rate_since_first_order'] = data.user_orders / (data.user_orders - data.up_first_order + 1)

    ############################

    user_dep_stat = pd.read_pickle('data/user_department_products.pkl')
    user_aisle_stat = pd.read_pickle('data/user_aisle_products.pkl')

    order_train = pd.merge(order_train, products, on='product_id')
    order_train = pd.merge(order_train, orders, on='order_id')
    order_train = pd.merge(order_train, user_dep_stat, on=['user_id', 'department_id'])
    order_train = pd.merge(order_train, user_aisle_stat, on=['user_id', 'aisle_id'])

    order_train = pd.merge(order_train, prod_usr, on='product_id')
    order_train = pd.merge(order_train, prod_usr_reordered, on='product_id', how='left')
    order_train.prod_users_unq_reordered.fillna(0, inplace=True)

    order_train = pd.merge(order_train, data, on=['product_id', 'user_id'])
    order_train = pd.merge(order_train, data_dow, on=['product_id', 'user_id', 'order_dow'], how='left')

    order_train['aisle_reordered_ratio'] = order_train.aisle_reordered / order_train.user_orders
    order_train['dep_reordered_ratio'] = order_train.dep_reordered / order_train.user_orders

    order_train = pd.merge(order_train, product_periods, on=['user_id',  'product_id'])
    order_train = pd.merge(order_train, product_embeddings, on=['product_id'])
    # order_train = pd.merge(order_train, weights, on='order_id')

    # order_train = pd.merge(order_train, order_prev, on=['order_id', 'product_id'], how='left')
    # order_train.reordered_prev = order_train.reordered_prev.astype(np.float32) + 1.
    # order_train['reordered_prev'].fillna(0, inplace=True)
    # order_train[['add_to_cart_order_prev', 'order_dow_prev', 'order_hour_of_day_prev']].fillna(255, inplace=True)

    print('data is joined')

    # order_train.days_since_prior_order_mean -= order_train.days_since_prior_order
    # order_train.days_since_prior_order_median -= order_train.days_since_prior_order
    #
    # order_train.order_dow_mean -= order_train.order_dow
    # order_train.order_dow_median -= order_train.order_dow
    #
    # order_train.order_hour_of_day_mean -= order_train.order_hour_of_day
    # order_train.order_hour_of_day_median -= order_train.order_hour_of_day

    unique_orders = np.unique(order_train.order_id)
    orders_train, orders_test = train_test_split(unique_orders, test_size=0.25, random_state=2017)

    order_test = order_train.loc[np.in1d(order_train.order_id, orders_test)]
    order_train = order_train.loc[np.in1d(order_train.order_id, orders_train)]

    features = [
        # 'reordered_dow_ration', 'reordered_dow', 'reordered_dow_size',
        # 'reordered_prev', 'add_to_cart_order_prev', 'order_dow_prev', 'order_hour_of_day_prev',
        'user_product_reordered_ratio', 'reordered_sum',
        'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
        'reorder_prob',
        'last', 'prev1', 'prev2', 'median', 'mean',
        'dep_reordered_ratio', 'aisle_reordered_ratio',
        'aisle_products',
        'aisle_reordered',
        'dep_products',
        'dep_reordered',
        'prod_users_unq', 'prod_users_unq_reordered',
        'order_number', 'prod_add_to_card_mean',
                'days_since_prior_order',
        'order_dow', 'order_hour_of_day',
                'reorder_ration',
                        'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
        # 'user_median_days_since_prior',
                        'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
                        'prod_orders', 'prod_reorders',
                        'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
                        'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
        # 'up_median_cart_position',
                             'days_since_prior_order_mean',
        # 'days_since_prior_order_median',
        'order_dow_mean',
        # 'order_dow_median',
        #                      'order_hour_of_day_mean',
        # 'order_hour_of_day_median'
                ]
    features.extend(embedings)

    print('not included', set(order_train.columns.tolist()) - set(features))

    data = order_train[features].fillna(-1.).values.astype(np.float32)

    data_categoties = order_train[['product_id', 'aisle_id', 'department_id']].values.astype(np.uint32)
    labels = order_train[['reordered']].values.astype(np.float32).flatten()
    # weights = order_train.weights.values.astype(np.float32)
    # weights = 1./np.maximum(weights, 1.0)

    data_val = order_test[features].fillna(-1.).values.astype(np.float32)

    data_categoties_val = order_test[['product_id', 'aisle_id', 'department_id']].values.astype(np.uint32)
    labels_val = order_test[['reordered']].values.astype(np.float32).flatten()

    print(data.shape, data_categoties.shape, labels.shape)

    # assert data.shape[0] == 8474661



    config = json.dumps({'objective': 1,
                         'internals':
                             {
                                 'compute_overlap': 3,
                                 'double_precision': True,
                                 'seed': 2017
                             },
                         'verbose':
                             {
                                 'gpu': True,
                                 'booster': True,
                                 'data': True
                             },
                         'tree':
                             {
                                 'eta': 0.01,
                                 'max_depth': 10,
                                 'gamma': 0.0,
                                 'min_child_weight':20.0,
                                 'min_leaf_size': 0,
                                 'colsample_bytree': 0.6,
                                 'colsample_bylevel': 0.6,
                                 'lambda': 0.1,
                                 'gamma_relative': 0.0001
                             }})

    print(config)

    data = arboretum.DMatrix(data, data_category=data_categoties, y=labels)
    data_val = arboretum.DMatrix(data_val, data_category=data_categoties_val)

    model = arboretum.Garden(config, data)

    print('training...')

    best_logloss = 1.0
    best_rocauc = 0

    best_iter_logloss = best_iter_rocauc = -1

    with ThreadPoolExecutor(max_workers=4) as executor:
        # grow trees
        for i in range(20000):
            print('tree', i)
            model.grow_tree()
            model.append_last_tree(data_val)
            if i % 5 == 0:
                pred = model.get_y(data)
                pred_val = model.get_y(data_val)
                logloss = executor.submit(sklearn.metrics.log_loss, labels, pred, eps=1e-6)
                logloss_val = executor.submit(sklearn.metrics.log_loss, labels_val, pred_val, eps=1e-6)
                rocauc = executor.submit(roc_auc_score, labels, pred)
                rocauc_val = executor.submit(roc_auc_score, labels_val, pred_val)
                # fscore_train = fscore(true_value_matrix, pred, order_index, product_index, len(orders_unique), len(products_unique), threshold=[0.16, 0.17, 0.18, 0.19, 0.2, 0.21])
                # fscore_value = fscore(true_value_matrix_val, pred_val, order_index_val, product_index_val, len(orders_unique_val), len(products_unique_val), threshold=[0.16, 0.17, 0.18, 0.19, 0.2, 0.21])
                logloss = logloss.result()
                logloss_val = logloss_val.result()
                rocauc = rocauc.result()
                rocauc_val = rocauc_val.result()
                print('train', logloss, rocauc,
                      'val', logloss_val, rocauc_val)
                if rocauc_val > best_rocauc:
                    print('best roc auc ', rocauc_val)
                    best_rocauc = rocauc_val
                    best_iter_rocauc = i
                if logloss_val < best_logloss:
                    print('best logloss', logloss_val)
                    best_logloss = logloss_val
                    best_iter_logloss = i

        print('best roc auc iteration', best_rocauc, best_iter_rocauc)
        print('best loggloss iteration', best_logloss, best_iter_logloss)