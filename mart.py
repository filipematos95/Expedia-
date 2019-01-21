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



metric = pyltr.metrics.NDCG(k=10)

all_features = ['srch_id', 'site_id', 'visitor_location_country_id',
   'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
   'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
   'prop_location_score1', 'prop_location_score2',
   'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
   'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
   'srch_adults_count', 'srch_children_count', 'srch_room_count',
   'srch_saturday_night_bool', 'srch_query_affinity_score',
   'orig_destination_distance', 'random_bool', 'click_bool',
   'gross_bookings_usd', 'booking_bool', 'rate_sum', 'inv_sum',
   'diff_mean', 'rate_abs', 'inv_abs']

col = ['rate_sum', 'inv_sum','prop_starrating', 'prop_review_score', 'prop_brand_bool',
    'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 
    'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
    'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
    'srch_adults_count', 'srch_children_count', 'srch_room_count']


filename = 'data/clean_train.csv'
nrows = 20000


def data_sets(filename, col, nrows):
    
    df = pd.read_csv(filename, skiprows=(1,2), nrows=nrows)
    Ty = df['click_bool'] + df['booking_bool']
    TX = df[col] 
    Tqids = df['srch_id']
    
    
    df2 = pd.read_csv(filename, skiprows=range(1,nrows + 2), nrows=nrows)
    Vy = df2['click_bool'] + df['booking_bool']
    VX = df2[col] 
    Vqids = df2['srch_id']
    
    df3 = pd.read_csv(filename, skiprows=range(1,2*nrows+2), nrows=nrows)
    Ey = df3['click_bool'] + df['booking_bool']
    EX = df3[col] 
    Eqids = df3['srch_id']
    
    return Ty, TX, Tqids, Vy, VX, Tqids, Ey, Ex, Eqids



Ty, TX, Tqids, Vy, VX, Vqids, Ey, Ex, Eqids = data_sets(filename, col, nrows)


VX = VX.fillna(0)
TX = TX.fillna(0)
EX = EX.fillna(0)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)


Epred = model.predict(EX)

print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))

