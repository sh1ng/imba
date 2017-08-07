* python3 first
* copy all *csv files into data folder
* install arboretum https://github.com/sh1ng/arboretum
* install lightgbm
* $ python3 create_products.py
* $ python3 split_data_set.py
* $ python3 orders_comsum.py
* $ python3 user_product_rank.py
* $ python3 create_prod2vec_dataset.py
* $ python3 skip_gram_train.py
* $ python3 skip_gram_get.py
* $ python3 arboretum_cv.py # optional just to see CV
* $ python3 lgbm_cv.py # optional...
* $ python3 arboretum_submition.py # prediction with arboretum
* $ python3 lgbm_submition.py # prediction with lgbm
* merge probabilities from 'data/prediction_arboretum.pkl' and 'data/prediction_lgbm.pkl'
* $ python3 f1_optimal.py 
* PROFIT!!!!!
