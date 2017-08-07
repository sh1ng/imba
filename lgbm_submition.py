import gc
import pandas as pd
import numpy as np
import os
import arboretum
import lightgbm as lgb
import json
import sklearn.metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.sparse import dok_matrix, coo_matrix
from sklearn.utils.multiclass import  type_of_target



def fscore(true_value_matrix, prediction, order_index, product_index, rows, cols, threshold=[0.5]):

    prediction_value_matrix = coo_matrix((prediction, (order_index, product_index)), shape=(rows, cols), dtype=np.float32)
    # prediction_value_matrix.eliminate_zeros()

    return list(map(lambda x: f1_score(true_value_matrix, prediction_value_matrix > x, average='samples'), threshold))


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

    product_embeddings = pd.read_pickle('data/product_embeddings.pkl')
    embedings = list(range(32))
    product_embeddings = product_embeddings[embedings + ['product_id']]

    order_train = pd.read_pickle(os.path.join(path, 'chunk_0.pkl'))
    order_test = order_train.loc[order_train.eval_set == "test", ['order_id', 'product_id']]
    order_train = order_train.loc[order_train.eval_set == "train", ['order_id',  'product_id',  'reordered']]

    product_periods = pd.read_pickle(os.path.join(path, 'product_periods_stat.pkl')).fillna(9999)

    print(order_train.columns)

    ###########################

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

    order_stat = orders_products.groupby('order_id').agg({'order_id': 'size'}) \
        .rename(columns={'order_id': 'order_size'}).reset_index()

    orders_products = pd.merge(orders_products, order_stat, on='order_id')
    orders_products['add_to_cart_order_inverted'] = orders_products.order_size - orders_products.add_to_cart_order
    orders_products['add_to_cart_order_relative'] = orders_products.add_to_cart_order / orders_products.order_size

    data = orders_products.groupby(['user_id', 'product_id']).agg({'user_id': 'size',
                                                                   'order_number': ['min', 'max'],
                                                                   'add_to_cart_order': ['mean', 'median'],
                                                                   'days_since_prior_order': ['mean', 'median'],
                                                                   'order_dow': ['mean', 'median'],
                                                                   'order_hour_of_day': ['mean', 'median'],
                                                                   'add_to_cart_order_inverted': ['mean', 'median'],
                                                                   'add_to_cart_order_relative': ['mean', 'median'],
                                                                   'reordered': ['sum']})

    data.columns = data.columns.droplevel(0)
    data.columns = ['up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position', 'up_median_cart_position',
                    'days_since_prior_order_mean', 'days_since_prior_order_median', 'order_dow_mean',
                    'order_dow_median',
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

    ############### train

    print(order_train.shape)
    order_train = pd.merge(order_train, products, on='product_id')
    print(order_train.shape)
    order_train = pd.merge(order_train, orders, on='order_id')
    print(order_train.shape)
    order_train = pd.merge(order_train, user_dep_stat, on=['user_id', 'department_id'])
    print(order_train.shape)
    order_train = pd.merge(order_train, user_aisle_stat, on=['user_id', 'aisle_id'])
    print(order_train.shape)

    order_train = pd.merge(order_train, prod_usr, on='product_id')
    print(order_train.shape)
    order_train = pd.merge(order_train, prod_usr_reordered, on='product_id', how='left')
    order_train.prod_users_unq_reordered.fillna(0, inplace=True)
    print(order_train.shape)

    order_train = pd.merge(order_train, data, on=['product_id', 'user_id'])
    print(order_train.shape)

    order_train['aisle_reordered_ratio'] = order_train.aisle_reordered / order_train.user_orders
    order_train['dep_reordered_ratio'] = order_train.dep_reordered / order_train.user_orders

    order_train = pd.merge(order_train, product_periods, on=['user_id',  'product_id'])

    ##############

    order_test = pd.merge(order_test, products, on='product_id')
    order_test = pd.merge(order_test, orders, on='order_id')
    order_test = pd.merge(order_test, user_dep_stat, on=['user_id', 'department_id'])
    order_test = pd.merge(order_test, user_aisle_stat, on=['user_id', 'aisle_id'])

    order_test = pd.merge(order_test, prod_usr, on='product_id')
    order_test = pd.merge(order_test, prod_usr_reordered, on='product_id', how='left')
    order_train.prod_users_unq_reordered.fillna(0, inplace=True)

    order_test = pd.merge(order_test, data, on=['product_id', 'user_id'])

    order_test['aisle_reordered_ratio'] = order_test.aisle_reordered / order_test.user_orders
    order_test['dep_reordered_ratio'] = order_test.dep_reordered / order_test.user_orders

    order_test = pd.merge(order_test, product_periods, on=['user_id', 'product_id'])

    order_train = pd.merge(order_train, product_embeddings, on=['product_id'])
    order_test = pd.merge(order_test, product_embeddings, on=['product_id'])

    print('data is joined')

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
        'order_hour_of_day_mean',
        # 'order_hour_of_day_median'
    ]
    features.extend(embedings)
    categories = ['product_id', 'aisle_id', 'department_id']
    features.extend(embedings)
    cat_features = ','.join(map(lambda x: str(x + len(features)), range(len(categories))))
    features.extend(categories)

    print('not included', set(order_train.columns.tolist()) - set(features))

    data = order_train[features]
    labels = order_train[['reordered']].values.astype(np.float32).flatten()

    data_val = order_test[features]

    assert data.shape[0] == 8474661

    lgb_train = lgb.Dataset(data, labels, categorical_feature=cat_features)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=380)

    prediction = gbm.predict(data_val)
    # prediction = model.predict(data_val)
    orders = order_test.order_id.values
    products = order_test.product_id.values

    result = pd.DataFrame({'product_id': products, 'order_id': orders, 'prediction': prediction})
    result.to_pickle('data/prediction_lgbm.pkl')
