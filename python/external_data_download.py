#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import pandas as pd
import numpy as np
import os
import requests
from tqdm import tqdm
from io import StringIO
from constants import BJ_STATION_CHN_PINGYING_MAPPING, LONDON_STATIONS, LONDON_API_COL_MAPPING, BEIJING_STATIONS
import time
from util import normalize_datetime

# BEIJING data from external api
# retrive historical beijing aq data from http://beijingair.sinaapp.com/
# data source posted in forum: https://biendata.com/forum/view_post_category/94
# LONDON data from external api
# I posted on forum: https://biendata.com/forum/view_post_category/80

def main():
    os.system('mkdir -p ../input/bj_pm_his')
    os.system('mkdir -p ../input/bj_o3_his')
    currentdate = pd.to_datetime('2014-03-31')
    enddate = pd.to_datetime('2018-04-03')
    bj_his_pm_data, bj_his_o3_data = [], []
    pm_failed_dates, o3_failed_dates = [], []
    while currentdate < enddate:
        date_str = '{}{:02}{:02}'.format(currentdate.year, currentdate.month, currentdate.day)
        print("# ---- trying to download data on date {}".format(date_str))
        pm_url = 'http://beijingair.sinaapp.com/data/beijing/all/{}/csv'.format(date_str)
        o3_url = 'http://beijingair.sinaapp.com/data/beijing/extra/{}/csv'.format(date_str)
        pm_data_path = '../input/bj_pm_his/{}.csv'.format(date_str)
        if os.path.exists(pm_data_path):
            data = pd.read_csv(pm_data_path)
            bj_his_pm_data.append(data)
        else:
            try:
                response = requests.get(pm_url, timeout=3)
                if response.status_code == 200:
                    data = pd.read_csv(StringIO(response.text))
                    data.columns = [col
                                    if col not in BJ_STATION_CHN_PINGYING_MAPPING
                                    else BJ_STATION_CHN_PINGYING_MAPPING[col]
                                    for col in data.columns
                                    ]
                    data.to_csv(pm_data_path, index=False)
                    bj_his_pm_data.append(data)
                else:
                    print('response code != 200')
                    pm_failed_dates.append(currentdate)
            except:
                print("# not a perticular meaningful printout")
                pm_failed_dates.append(currentdate)
        o3_data_path = '../input/bj_o3_his/{}.csv'.format(date_str)
        if os.path.exists(o3_data_path):
            data = pd.read_csv(o3_data_path)
            bj_his_o3_data.append(data)
        else:
            try:
                response = requests.get(o3_url, timeout=3)
                if response.status_code == 200:
                    data = pd.read_csv(StringIO(response.text))
                    data.columns = [col
                                    if col not in BJ_STATION_CHN_PINGYING_MAPPING
                                    else BJ_STATION_CHN_PINGYING_MAPPING[col]
                                    for col in data.columns
                                    ]
                    data.to_csv(o3_data_path, index=False)
                    bj_his_o3_data.append(data)
                else:
                    print('response code != 200')
                    o3_failed_dates.append(currentdate)
            except:
                print("# not a perticular meaningful printout")
                o3_failed_dates.append(currentdate)
        time.sleep(1)
        currentdate += pd.Timedelta(1, unit='D')

    bj_his_data = pd.concat(bj_his_pm_data + bj_his_o3_data, axis=0)
    bj_his_data['utc_time'] = bj_his_data.\
        apply(lambda row: str(int(row['date'])) + ' {:02}:00:00'.format(row['hour']), axis=1)
    bj_his_data['utc_time'] = pd.to_datetime(bj_his_data['utc_time'])
    bj_his_data['utc_time'] = bj_his_data['utc_time'].map(lambda x: x - pd.Timedelta(8, unit='h'))
    bj_his_data = bj_his_data.loc[bj_his_data['type'].isin(['PM2.5', 'PM10', 'O3'])]
    bj_his_data.drop(labels=['date', 'hour'], axis=1, inplace=True)
    reshape_aq_data = []
    for station in BEIJING_STATIONS:
        s_aq_data = bj_his_data[[station] + ['utc_time', 'type']].copy()
        s_aq_data.columns = ['values', 'utc_time', 'type']
        d = s_aq_data.pivot(index='utc_time', columns='type', values='values').reset_index()
        d = pd.DataFrame(d.values, columns=d.columns.values)
        d['stationId'] = station
        d = d[['stationId', 'utc_time', 'PM2.5', 'PM10', 'O3']]
        reshape_aq_data.append(d)
    bj_his_data = pd.concat(reshape_aq_data, axis=0)
    bj_his_data['utc_time'] = pd.to_datetime(bj_his_data['utc_time'])
    bj_his_data.drop_duplicates(subset=['stationId', 'utc_time'], keep='last', inplace=True)
    bj_his_data = bj_his_data.loc[bj_his_data['utc_time'] >= pd.to_datetime('2014-03-31')]
    bj_his_data = bj_his_data.loc[bj_his_data['utc_time'] < pd.to_datetime('2018-04-01')]
    bj_his_data = normalize_datetime(bj_his_data)
    bj_his_data.to_csv('../input/bj_api_his.csv', index=False)

    bj_biendata = pd.read_csv(filepath_or_buffer='../input/beijing_history.csv', parse_dates=['utc_time'])
    bj_joined = pd.merge(left=bj_his_data, right=bj_biendata, how='left', on=['stationId', 'utc_time'],
                         suffixes=['alternative', 'biendata'])
    for i in tqdm(range(bj_joined.shape[0])):
        if pd.isnull(bj_joined.ix[i, 2]) and not pd.isnull(bj_joined.ix[i, 5]):
            bj_joined.ix[i, 2] = bj_joined.ix[i, 5]
        if pd.isnull(bj_joined.ix[i, 3]) and not pd.isnull(bj_joined.ix[i, 6]):
            bj_joined.ix[i, 3] = bj_joined.ix[i, 6]
        if pd.isnull(bj_joined.ix[i, 4]) and not pd.isnull(bj_joined.ix[i, 7]):
            bj_joined.ix[i, 4] = bj_joined.ix[i, 7]

    bj_filled = bj_joined[['stationId', 'utc_time', 'PM2.5alternative', 'PM10alternative', 'O3alternative']]
    bj_filled.columns = ['stationId', 'utc_time', 'PM2.5', 'PM10', 'O3']
    bj_filled.to_csv('../input/bj_api_his.csv', index=False)

    os.system('mkdir -p ../input/ld_his')
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    years = [2014, 2015, 2016, 2017, 2018]
    dates = ['01' + month + str(year) for year in years for month in months]
    dates = dates[:-7]
    ld_his_data = []
    ld_failed = []

    for start, end in zip(dates[:-1], dates[1:]):
        for station in LONDON_STATIONS:
            datapath = '../input/ld_his/{}_{}-{}.csv'.format(station, start, end)
            url = 'http://api.erg.kcl.ac.uk/AirQuality/Data/Site/Wide/SiteCode={}'.format(station) + \
                  '/StartDate={}/EndDate={}/csv'.format(start, end)
            if os.path.exists(datapath):
                data = pd.read_csv(datapath)
                ld_his_data.append(data)
            else:
                print('# ---- trying to get data from :' + url)
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = pd.read_csv(StringIO(response.text))
                        data.columns = pd.Series(data.columns).map(lambda x: x.split(': ')[-1])
                        data.columns = [col
                                        if col not in LONDON_API_COL_MAPPING
                                        else LONDON_API_COL_MAPPING[col]
                                        for col in data.columns
                                        ]
                        data = data[['utc_time', 'PM2.5', 'PM10']]
                        data['stationId'] = station
                        ld_his_data.append(data)
                        data.to_csv('../input/ld_his/{}_{}-{}.csv'.format(station, start, end))
                    else:
                        print('response code != 200')
                        ld_failed.append('{}_{}-{}'.format(station, start, end))
                except:
                    print("# not a perticular meaningful printout")
                    ld_failed.append('{}_{}-{}'.format(station, start, end))
    ld_df = pd.concat(ld_his_data)
    ld_df['utc_time'] = pd.to_datetime(ld_df['utc_time'])
    ld_df = ld_df.loc[ld_df['utc_time'] <= '2018-04-02']
    ld_df = ld_df[['stationId', 'utc_time', 'PM2.5', 'PM10']]
    start, end = ld_df.utc_time.min(), ld_df.utc_time.max()
    print('min max datetime are: {} {}'.format(start, end))
    ld_df['utc_time'] = pd.to_datetime(ld_df['utc_time'])
    ld_df.drop_duplicates(subset=['stationId', 'utc_time'], keep='last', inplace=True)
    ld_df = ld_df.loc[ld_df['utc_time'] >= pd.to_datetime('2014-03-31 00:00:00')]
    ld_df = ld_df.loc[ld_df['utc_time'] <= pd.to_datetime('2018-03-31 23:00:00')]
    ld_df = normalize_datetime(ld_df)
    ld_df.sort_values(['stationId', 'utc_time'], inplace=True)
    ld_df['O3'] = np.nan
    ld_df.to_csv('../input/ld_api_his.csv', index=False)


if __name__ == '__main__':
    main()
