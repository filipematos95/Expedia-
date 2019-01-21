#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#                LambdaMART             #
#                                       #
#########################################

#                   Credits pyltr
#               
#           source: https://github.com/jma127/pyltr
#
#           Copyright (c) 2015, Jerry Ma
#               All rights reserved.

import pyltr
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os 

# with open('data/ex/train.txt') as trainfile, \
#         open('data/ex/vali.txt') as valifile, \
#         open('data/ex/test.txt') as evalfile:
#     TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
#     VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
#     EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)



metric = pyltr.metrics.NDCG(k=10)



# this
col = ['hotel_quality_1', 'price_usd_med', 'prop_id', 'hotel_quality_2',
    'score2ma', 'score1d2', 'price_usd', 'total_fee', 'ump', 'prop_location_score2',
    'promotion_flag_mean', 'price_usd_mean', 'per_fee', 'prop_log_historical_price',
    'price_diff', 'promotion_flag', 'rate_sum', 'prop_log_historical_price_med',
    'prop_country_id', 'starrating_diff', 'prop_location_score2_mean']


nrows = 20000
train = pd.read_csv('train.csv', skiprows=(1,2), nrows=nrows)
list(train.columns)

filename = 'train.csv'

# Save model in new folder
def save(model, foldername = "\sample"):
    try:
        pathfile = os.path.join(os.getcwd(), foldername)
        os.makedirs(pathfile)
        
    except:
        print("iets anders")
    with open(os.path.join(pathfile, 'model'), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(model, output)

# save(model_1, "test2")

# Load model from folder
def load(foldername):
    
    directory = os.path.join(os.getcwd(), foldername)
    with open(os.path.join(directory, 'model'), 'rb') as input:
        model = pickle.load(input)
        
    return model
    
# x = load('test2')

def data_sets(filename, col, nrows, k=10):
    
    data = pd.read_csv(filename, skiprows=(1,2), nrows=nrows)
    index = pd.DataFrame(data['srch_id'].unique()).sample(n = len(data['srch_id'].unique()))
    index1 = index[0: int(len(index)*(1/3.0) )]
    index2 = index[ int(len(index)*(1/3.0)) : int(len(index)*(2/3.0)) ]
    index3 = index[int(len(index)*(2/3.0)):]
    data['booking_bool'] = 5*data['booking_bool']
    data['calc'] = data[['click_bool','booking_bool']].apply(np.max, axis=1)
    data['k'] = data.groupby('srch_id')['srch_id'].transform(lambda x: np.size(x))
    data = data[data['k']>k].copy()
    
    df1 = data[data['srch_id'].isin( list(index1[0]) )]
    df2 = data[data['srch_id'].isin( list(index2[0]) )]
    df3 = data[data['srch_id'].isin( list(index3[0]) )]

    # make them equal size
    min_size = min(min(len(df1), len(df2)), len(df3))
    df1 = df1[0:min_size]
    df2 = df2[0:min_size]
    df3 = df3[0:min_size]
    
    Ty = df1['calc']
    TX = df1[col] 
    Tqids = df1['srch_id']

    Vy = df2['calc']
    VX = df2[col] 
    Vqids = df2['srch_id']
    
    Ey = df3['calc']
    EX = df3[col] 
    Eqids = df3['srch_id']
    
    return Ty, TX, Tqids, Vy, VX, Tqids, Ey, EX, Eqids


Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids = data_sets(filename, col, nrows, 15)

TX.fillna(TX.mean(),inplace = True)
VX.fillna(VX.mean(),inplace = True)
EX.fillna(EX.mean(),inplace = True)

TX.fillna(0,inplace = True)
VX.fillna(0,inplace = True)
EX.fillna(0,inplace = True)

Ty = np.array(Ty)
Vy = np.array(Vy)
Ey = np.array(Ey)
TX = np.array(TX)
VX = np.array(VX)
EX = np.array(EX)
Tqids = np.array(Tqids)
Vqids = np.array(Vqids)
Eqids = np.array(Eqids)


# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.19,
    max_features=0.7,
    query_subsample=1.0,
    max_leaf_nodes=5,
    min_samples_leaf=10,
    verbose=1,
)



model.fit(TX, Ty, Tqids, monitor=monitor)
Epred = model.predict(EX)


print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))










# see below for further checking
x.estimators_fitted_
x.estimators_ 
x.estimators_ = model.estimators_[0:10]
x.feature_importances_
x.train_score_ 
pd.DataFrame([model.feature_importances_, EX.columns]).T.sort_values(0, ascending = False)

# see score below

def compute(r):
    k = min(len(r), 10)
    return ndcg_at_k(r,k)

def dcg_at_k(r,k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

import sys
import math
import copy
import re

def get_max_ndcg(k, *ins):
    '''This is a function to get maxium value of DCG@k. That is the DCG@k of sorted ground truth list. '''
    #print ins
    l = [i for i in ins]
    l = copy.copy(l[0])
    l.sort(None,None,True)
    #print l
    max = 0.0
    for i in range(k):
        #print l[i]/math.log(i+2,2)
        max += (math.pow(2, l[i])-1)/math.log(i+2,2)
        #max += l[i]/math.log(i+2,2)
    return max


def get_ndcg(s, k):
    '''This is a function to get ndcg '''
    z = get_max_ndcg(k, s)
    dcg = 0.0
    for i in range(k):
        #print s[i]/math.log(i+2,2)
        dcg += (math.pow(2, s[i])-1)/math.log(i+2,2)
        #dcg += s[i]/math.log(i+2,2)
    if z ==0:
        z = 1;
    ndcg = dcg/z
    #print "Line:%s, NDCG@%d is %f with DCG = %f, z = %f"%(s, k, ndcg,dcg, z)
    return ndcg


X = pd.DataFrame([Epred, Eqids])
X = X.T
X. columns = ['prob', 'srch_id']
X['points'] = Ey

X_sort = X.sort_values(['srch_id', 'prob'],ascending=[True, False]) 
X_sort['score'] = X_sort.groupby('srch_id').apply(lambda x: compute(x.points.values))
X_sort[['score']].dropna().mean()
X_sort['score2'] = X_sort.groupby('srch_id').apply(lambda x: get_ndcg(list(x.points.values),len(k)))
X_sort[['score2']].dropna().mean()
score = pd.DataFrame([X_sort['score'].dropna(),X_sort['score2'].dropna()]).T


# col = [
#  'orig_destination_distance',
#  'price_usd',
#  'promotion_flag',
#  'prop_brand_bool',
#  'prop_country_id',
#  'prop_id',
#  'prop_location_score1',
#  'prop_location_score2',
#  'prop_log_historical_price',
#  'prop_review_score',
#  'prop_starrating',
#  'random_bool',
#  'rate_sum',
#  'inv_sum',
#  'diff_mean',
#  'rate_abs',
#  'inv_abs',
#  'prop_location_score1_mean',
#  'prop_location_score2_mean',
#  'prop_log_historical_price_mean',
#  'price_usd_mean',
#  'promotion_flag_mean',
#  'orig_destination_distance_mean',
#  'prop_location_score1_std',
#  'prop_location_score2_std',
#  'prop_log_historical_price_std',
#  'price_usd_std',
#  'promotion_flag_std',
#  'orig_destination_distance_std',
#  'prop_location_score1_med',
#  'prop_location_score2_med',
#  'prop_log_historical_price_med',
#  'price_usd_med',
#  'promotion_flag_med',
#  'orig_destination_distance_med',
#  'ump',
#  'price_diff',
#  'starrating_diff',
#  'per_fee',
#  'prop_starrating_mean',
#  'prop_starrating_std',
#  'prop_starrating_med',
#  'score2ma',
#  'total_fee',
#  'score1d2',
#  'hotel_quality_1',
#  'hotel_quality_2'] 


# all_features = ['booking_bool',
#  'click_bool',
#  'gross_bookings_usd',
#  'orig_destination_distance',
#  'position',
#  'price_usd',
#  'promotion_flag',
#  'prop_brand_bool',
#  'prop_country_id',
#  'prop_id',
#  'prop_location_score1',
#  'prop_location_score2',
#  'prop_log_historical_price',
#  'prop_review_score',
#  'prop_starrating',
#  'random_bool',
#  'site_id',
#  'srch_adults_count',
#  'srch_booking_window',
#  'srch_children_count',
#  'srch_destination_id',
#  'srch_id',
#  'srch_length_of_stay',
#  'srch_query_affinity_score',
#  'srch_room_count',
#  'srch_saturday_night_bool',
#  'visitor_hist_adr_usd',
#  'visitor_hist_starrating',
#  'visitor_location_country_id',
#  'rate_sum',
#  'inv_sum',
#  'diff_mean',
#  'rate_abs',
#  'inv_abs',
#  'visitor_hist_starrating_mean',
#  'prop_id_mean',
#  'visitor_hist_adr_usd_mean',
#  'prop_location_score1_mean',
#  'prop_location_score2_mean',
#  'prop_log_historical_price_mean',
#  'position_mean',
#  'price_usd_mean',
#  'promotion_flag_mean',
#  'srch_destination_id_mean',
#  'srch_length_of_stay_mean',
#  'srch_booking_window_mean',
#  'srch_adults_count_mean',
#  'srch_children_count_mean',
#  'srch_room_count_mean',
#  'srch_saturday_night_bool_mean',
#  'srch_query_affinity_score_mean',
#  'orig_destination_distance_mean',
#  'visitor_hist_starrating_std',
#  'prop_id_std',
#  'visitor_hist_adr_usd_std',
#  'prop_location_score1_std',
#  'prop_location_score2_std',
#  'prop_log_historical_price_std',
#  'position_std',
#  'price_usd_std',
#  'promotion_flag_std',
#  'srch_destination_id_std',
#  'srch_length_of_stay_std',
#  'srch_booking_window_std',
#  'srch_adults_count_std',
#  'srch_children_count_std',
#  'srch_room_count_std',
#  'srch_saturday_night_bool_std',
#  'srch_query_affinity_score_std',
#  'orig_destination_distance_std',
#  'visitor_hist_starrating_med',
#  'prop_id_med',
#  'visitor_hist_adr_usd_med',
#  'prop_location_score1_med',
#  'prop_location_score2_med',
#  'prop_log_historical_price_med',
#  'position_med',
#  'price_usd_med',
#  'promotion_flag_med',
#  'srch_destination_id_med',
#  'srch_length_of_stay_med',
#  'srch_booking_window_med',
#  'srch_adults_count_med',
#  'srch_children_count_med',
#  'srch_room_count_med',
#  'srch_saturday_night_bool_med',
#  'srch_query_affinity_score_med',
#  'orig_destination_distance_med',
#  'ump',
#  'price_diff',
#  'starrating_diff',
#  'per_fee',
#  'prop_starrating_mean',
#  'prop_starrating_std',
#  'prop_starrating_med',
#  'score2ma',
#  'total_fee',
#  'score1d2',
#  'hotel_quality_1',
#  'hotel_quality_2'] 
 
 # all_features = ['srch_id', 'site_id', 'visitor_location_country_id',
#   'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
#   'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
#   'prop_location_score1', 'prop_location_score2',
#   'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
#   'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
#   'srch_adults_count', 'srch_children_count', 'srch_room_count',
#   'srch_saturday_night_bool', 'srch_query_affinity_score',
#   'orig_destination_distance', 'random_bool', 'click_bool',
#   'gross_bookings_usd', 'booking_bool', 'rate_sum', 'inv_sum',
#   'diff_mean', 'rate_abs', 'inv_abs']

# col = ['rate_sum', 'inv_sum','prop_starrating', 'prop_review_score', 'prop_brand_bool',
#     'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 
#     'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
#     'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
#     'srch_adults_count', 'srch_children_count', 'srch_room_count']