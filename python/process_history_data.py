#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import pandas as pd
from util import download_aq_data, normalize_datetime, get_date, fwbwmean_impute
from constants import AQ_COL_NAMES
bj_his_1 = pd.read_csv('../input/beijing_17_18_aq.csv', parse_dates=['utc_time'])
bj_his_2 = pd.read_csv('../input/beijing_18_aq.csv', parse_dates=['utc_time'])
bj_his_3 = download_aq_data(city='bj', start_date='2018-03-31', start_hour='16', end_date='2018-03-31', end_hour='23',
                            save=True)
bj_his = pd.concat([bj_his_1, bj_his_2, bj_his_3], axis = 0)
start, end = bj_his.utc_time.min(), bj_his.utc_time.max()
bj_his = normalize_datetime(bj_his)
bj_his.drop_duplicates(subset=['stationId', 'utc_time'], keep='first', inplace=True)
bj_his.sort_values(['stationId', 'utc_time'], inplace=True)
bj_his = bj_his.loc[bj_his['utc_time'] >= '2017-01-02']
bj_his = bj_his[AQ_COL_NAMES]
bj_his.to_csv('../input/beijing_history.csv', index=False)

ld_his_1 = pd.read_csv('../input/london_17_18_aq.csv')
ld_his_1.columns = ['id', 'utc_time', 'stationId', 'PM2.5', 'PM10', 'NO2']
ld_his_1 = ld_his_1[['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2']]
ld_his_1['utc_time'] = pd.to_datetime(ld_his_1['utc_time'])
ld_his_3 = download_aq_data(city='ld', start_date='2018-03-31', start_hour='00', end_date='2018-03-31', end_hour='23',
                            save=True)
ld_his = pd.concat([ld_his_1, ld_his_3], axis = 0)
start, end = ld_his.utc_time.min(), ld_his.utc_time.max()
ld_his = normalize_datetime(ld_his)
ld_his.drop_duplicates(subset=['stationId', 'utc_time'], keep='first', inplace=True)
ld_his.sort_values(['stationId', 'utc_time'], inplace=True)
ld_his = ld_his.loc[ld_his['utc_time'] >= '2017-01-02']
ld_his = ld_his[AQ_COL_NAMES]
ld_his.to_csv('../input/london_history.csv', index=False)