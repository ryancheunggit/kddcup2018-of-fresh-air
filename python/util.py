#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import time
import pandas as pd
import numpy as np
import requests
import argparse
from io import StringIO
from constants import (LONDON_STATIONS, BEIJING_STATIONS, BJ_STATION_CHN_PINGYING_MAPPING, LONDON_API_COL_MAPPING,
                       AQ_COL_NAMES, DATA_API_TOKEN, SUB_TOKEN, USERNAME, MONTH_DIGIT_STR_MAPPING
                       )
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
import pickle
from datetime import datetime


class StandardScaler():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values):
        self.mean = np.nanmean(values)
        self.std = np.nanstd(values)
        return self

    def transform(self, values):
        return (np.array(values)-self.mean) / self.std

    def inverse_transform(self, values):
        return self.mean + self.std * np.array(values)

    def fit_transform(self, values):
        self.mean = np.nanmean(values)
        self.std = np.nanstd(values)
        return self.transform(values)

    def mean_(self):
        assert self.mean is not None, 'not fitted yet'
        return self.mean

    def std_(self):
        assert self.std is not None, 'not fitted yet'
        return self.std

    def __repr__(self):
        return 'scalar with mean {:10.4f} and std {:10.4f}'.format(self.mean, self.std)


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    https://github.com/facebook/prophet/issues/223
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def str2bool(v):
    """
    used to parse string argument true and false to bool in argparse
    # https://stackoverflow.com/questions/15008758/
    # parsing-boolean-values-with-argparse
    """
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


@contextmanager
def timer(description):
    """ time the run time of the chunk of code """
    t0 = time.time()
    yield
    print(f'[{description}] done in {time.time() - t0:.0f} s')


def get_verbose_print(verbose_level):
    if verbose_level > 0:
        def vprint(*vargs):
            if verbose_level >= vargs[0]:
                print(vargs[1])
    else:
        def vprint(*vargs):
            pass
    setattr(vprint, 'verbose_level', verbose_level)
    return vprint


def get_date(datetime_obj):
    return str(datetime_obj).split()[0]


def lgbm_impute(data: pd.DataFrame, city: str, vprint=print) -> pd.DataFrame:
    assert city in ['bj', 'ld'], 'invalid city'
    assert os.path.exists('../models/{}_lgbm.pkl'.format(city)), 'lgb model not trained yet'

    vprint(1, 'impute data with lgbm')
    with suppress_stdout_stderr():
        lgb_models = pickle.load(open('../models/{}_lgbm.pkl'.format(city), 'rb'))

    if city == 'bj':
        measures, stations, threshold = ['PM2.5', 'PM10', 'O3'], BEIJING_STATIONS, 17
    else:
        measures, stations, threshold = ['PM2.5', 'PM10'], LONDON_STATIONS, 6

    dfs = {}
    for measure in measures:
        x_data = [data.loc[data.stationId == station, measure].values for station in stations]
        x_data = pd.DataFrame(np.array(x_data)).T
        x_data.columns = stations
        dfs[measure] = x_data.copy()

    for station in stations:
        for measure in measures:
            vprint(2, "impute {} {} - {} with lgb".format(city, station, measure))
            value = dfs[measure][station].copy()
            condition = value.isnull() & (dfs[measure].isnull().sum(axis=1) < threshold)
            predicted = lgb_models['{}-{}'.format(station, measure)].\
                predict(dfs[measure].loc[condition, [col for col in stations if col != station]])
            value[condition] = predicted
            data.loc[data['stationId'] == station, measure] = value.tolist()
    return data


def forward_backward_impute(data: pd.DataFrame, method: str) -> pd.DataFrame:
    assert method in ['day', 'hour', 'fwbw_day', 'fwbw_hour'], 'method of grouping not correct'
    df = data.copy()  # type: pd.DataFrame
    if method == 'day' or method == 'fwbw_day':
        df.loc[:, 'key'] = df['utc_time'].map(get_date)
    if method == 'hour' or method == 'fwbw_hour':
        df.loc[:, 'key'] = df['utc_time'].map(lambda x: str(x.hour))
    df.sort_values(by=['stationId', 'key'], axis=0, inplace=True)
    df = df.groupby(['stationId', 'key'], sort=False).apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    df.drop(labels='key', axis=1, inplace=True)
    df.sort_values(by=['stationId', 'utc_time'], axis=0, inplace=True)
    return df


def mean_impute(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()  # type: pd.DataFrame
    df = df.groupby('stationId', sort=False).apply(lambda x: x.fillna(x.mean())).reset_index(drop=True)
    return df


def impute(data: pd.DataFrame, lgbm: bool=False, day: bool=False, hour: bool=False, mean: bool=False, zero: bool=False,
           vprint=print) -> pd.DataFrame:
    df = data.copy()  # type: pd.DataFrame
    if lgbm:
        df.loc[df['stationId'].isin(BEIJING_STATIONS)] = lgbm_impute(
            data=df.loc[df['stationId'].isin(BEIJING_STATIONS)], city='bj', vprint=vprint)
        df.loc[df['stationId'].isin(LONDON_STATIONS)] = lgbm_impute(
            data=df.loc[df['stationId'].isin(LONDON_STATIONS)], city='ld', vprint=vprint)
    if day:
        vprint(2, 'day impute')
        df = forward_backward_impute(data=df, method='day')
    if hour:
        vprint(2, 'hour impute')
        df = forward_backward_impute(data=df, method='hour')
    if mean:
        vprint(2, 'mean impute')
        df = mean_impute(data=df)
    if zero:
        vprint(2, 'zero impute')
        df.fillna(0, inplace=True)
    return df


def fwbwmean_impute(data: pd.DataFrame, main_method: str='mean', residual_method: str='mean'):
    # TODO: use above one instead
    df = data.copy()  # type: pd.DataFrame
    if main_method == 'fwbw_day' or main_method == 'day':
        df = impute(df, day=True)
    if main_method == 'fwbw_hour' or main_method == 'hour':
        df = impute(df, hour=True)
    if main_method == 'mean' or residual_method == 'mean':
        df = impute(df, mean=True)
    if residual_method == 'zero':
        df = impute(df, zero=True)
    return df


def smape(y_true, y_pred):
    """ https://biendata.com/competition/kdd_2018/evaluation/ """
    denom = (y_true + y_pred) / 2
    ape = np.abs(y_true - y_pred) / denom
    ape[denom == 0] = 0
    return np.nanmean(ape)


def official_smape(actual, predicted):
    """ https://biendata.com/competition/kdd_2018/evaluation/ """
    dividend = np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator != 0, casting='unsafe'))


def fix_nat(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()  # type: pd.DataFrame
    df['PM2.5'] = df['PM2.5'].map(lambda x: x if str(x) != 'NaT' else np.nan)
    df['PM10'] = df['PM10'].map(lambda x: x if str(x) != 'NaT' else np.nan)
    df['O3'] = df['O3'].map(lambda x: x if str(x) != 'NaT' else np.nan)
    return df


def long_to_wide(data, index='stationId', columns='utc_time'):
    df = data.copy()  # type: pd.DataFrame
    pm25data = df.pivot(index=index, columns=columns, values='PM2.5').reset_index()
    pm25data.stationId = pm25data.stationId.map(lambda x: x + '#PM2.5#')
    pm10data = df.pivot(index=index, columns=columns, values='PM10').reset_index()
    pm10data.stationId = pm10data.stationId.map(lambda x: x + '#PM10#')
    o3data = df.loc[~df.stationId.isin(LONDON_STATIONS)].pivot(index=index, columns=columns, values='O3').reset_index()
    o3data.stationId = o3data.stationId.map(lambda x: x + '#O3#')
    wdata = pd.concat([pm25data, pm10data, o3data], axis=0)
    return wdata


def wide_fw_x_y_split(wdata, history_length, split_date, for_prediction=False):
    col_indices = wdata.columns.tolist()
    pred_start = split_date + pd.Timedelta(1, unit='D')
    pred_end = pred_start + pd.Timedelta(47, unit='h')
    his_end = split_date - pd.Timedelta(1, unit='h')
    his_start = split_date - pd.Timedelta(history_length, unit='D')
    x = wdata.iloc[:, [0] + list(range(col_indices.index(his_start), col_indices.index(his_end) + 1))].copy()
    y = None
    if not for_prediction:
        y = wdata.iloc[:, [0] + list(range(col_indices.index(pred_start), col_indices.index(pred_end) + 1))].copy()
    return x, y


def extract_median(ws, ldata, split_date):
    window = ldata.loc[(ldata.utc_time >= split_date - pd.Timedelta(ws, unit='D')) & (ldata.utc_time < split_date),
                       AQ_COL_NAMES].copy()
    window['hour'] = window['utc_time'].map(lambda x: x.hour)
    medians = window. \
        groupby(['stationId', 'hour']). \
        agg({'PM2.5': np.median, 'PM10': np.median, 'O3': np.median}). \
        reset_index()
    medians = long_to_wide(medians, index='stationId', columns='hour')
    return medians


def extract_median_features(ldata, window_sizes, split_date, n_thread):
    extract_median_thread = partial(extract_median, ldata=ldata, split_date=split_date)
    pool = Pool(n_thread)
    list_of_medians = pool.map(extract_median_thread, window_sizes)
    pool.terminate()
    pool.close()
    median_features = list_of_medians[0]
    for window_idx, median_df in enumerate(list_of_medians[1:]):
        median_features = pd.merge(left=median_features, right=median_df, on='stationId',
                                   suffixes=['', 'mf_window_{}'.format(window_idx)])
    return median_features


def wide_make_fw_x_y(wdata, history_length, split_date, num_shifts=1, shift_step=1, use_medians=False, ldata=None,
                     window_sizes=None, median_shifts=None, use_indicators=False, for_prediction=False, n_thread=1,
                     vprint=print, save_feature=True, use_cache=True, window_name=None):
    x_data, y_data = [], []
    split_date = pd.to_datetime(split_date)
    for shift in range(num_shifts):
        feature_name = 'his_{}_spl_{}_med_{}_medshf_{}_fp_{}'.format(history_length, get_date(split_date), use_medians,
                                                                     median_shifts, for_prediction)
        if window_name is not None:
            feature_name += '_{}'.format(window_name)
        if os.path.exists('../features/{}.pkl'.format(feature_name)) and use_cache:
            vprint(2, "loading pickled data")
            x, y = pickle.load(open('../features/{}.pkl'.format(feature_name), 'rb'))
            x_data.append(x)
            y_data.append(y)
        else:
            vprint(2, '# ---- making the {}th shift ----'.format(shift + 1))
            x, y = wide_fw_x_y_split(wdata=wdata, history_length=history_length, split_date=split_date,
                                     for_prediction=for_prediction)
            vprint(1, 'X: split {} shift {} from {} to {}'.format(split_date, shift, x.columns[1], x.columns[-1]))
            x.columns = ['stationId'] + ['d{}_h{}'.format(d, h) for d in range(history_length) for h in range(24)]
            if use_medians:
                assert (ldata is not None) and (window_sizes is not None), 'need provide window_sizes and ldata'
                for median_shift in range(median_shifts):
                    x_medians = extract_median_features(ldata=ldata, window_sizes=window_sizes,
                                                        split_date=split_date - pd.Timedelta(median_shift, unit='D'),
                                                        n_thread=n_thread)
                    x = pd.merge(left=x, right=x_medians, on='stationId', how='left')

            x['stationId'] = x['stationId'].map(lambda d: d + str(split_date).split(' ')[0])
            x_data.append(x)

            if not for_prediction:
                y['stationId'] = y['stationId'].map(lambda x: x + str(split_date).split(' ')[0])
                vprint(1, 'y: split {} shift {} from {} to {}'.format(split_date, shift, y.columns[1], y.columns[-1]))
                y.columns = ['stationId'] + ['d{}_h{}'.format(d, h)
                                             for d in [i + history_length for i in [1, 2]]
                                             for h in range(24)]
                y_data.append(y)

            if save_feature:
                vprint(2, 'dump feature to pickle file')
                os.system('mkdir -p ../features')
                with open('../features/{}.pkl'.format(feature_name), 'wb') as f:
                    pickle.dump([x, y], f)
        split_date -= pd.Timedelta(shift_step, unit='D')

    x_data = pd.concat(x_data, axis=0)
    y_data = pd.concat(y_data, axis=0) if not for_prediction else None

    if use_indicators:
        x_data['station'] = x_data['stationId'].map(lambda x: x.split('#')[0])
        x_data['measure'] = x_data['stationId'].map(lambda x: x.split('#')[1])
        x_data['city'] = x_data['station'].map(lambda x: 1 if x in LONDON_STATIONS else 0)
        x_measure_dummies = pd.get_dummies(x_data['measure'])
        x_data = pd.concat(objs=[x_data, x_measure_dummies],axis=1)
        x_data.drop(labels=['station', 'measure'], axis=1, inplace=True)
    return x_data, y_data


def standardize_data(data: pd.DataFrame, vprint=print):
    vprint(2, '# ---- standardizing data to zero mean unit variance')
    df = data.copy()  # type: pd.DataFrame
    scalers = {}
    stations = df.stationId.map(lambda x: x.split('#')[0]).unique()
    for station in stations:
        measures = ['PM2.5', 'PM10', 'O3']
        if station in LONDON_STATIONS:
            measures.remove('O3')
        for measure in measures:
            key = '#'.join([station, measure])
            values = df.loc[df['stationId'].map(lambda x: x.split('#')[0] == station), measure]
            scalers[key] = StandardScaler().fit(values)
            df.loc[df['stationId'].map(lambda x: x.split('#')[0] == station), measure] = scalers[key].transform(values)
    return df, scalers


def normalize_datetime(df, keys=('stationId', 'utc_time'), min_time=None, max_time=None):
    assert all(key in df.columns for key in keys), 'some key not in columns'

    if not min_time:
        min_time = df.utc_time.min()
    if not max_time:
        max_time = df.utc_time.max()
    df.drop_duplicates(subset=keys, keep='first', inplace=True)
    stations = df.stationId.unique()
    utc_times = []
    current = min_time
    while current <= max_time:
        utc_times.append(current)
        current += pd.Timedelta(1, 'h')
    left = pd.DataFrame(pd.Series([s, t]) for s in stations for t in utc_times)
    left.columns = ['stationId', 'utc_time']
    if left.shape[0] != df.shape[0]:
        df = pd.merge(left=left, right=df, on=['stationId', 'utc_time'], how='left')
    df.sort_values(by=['stationId', 'utc_time'], inplace=True)
    return df


def get_date_col_index(dataframe, day):
    idx = dataframe.columns.tolist().index(day)
    if idx != len(dataframe.columns):
        return idx
    return None


def download_aq_data(city, start_date, start_hour, end_date, end_hour, save=False, partial_data=False,
                     data_source='biendata', vprint=print):
    assert city in ['bj', 'ld'], 'invalid city'
    start_time = '{}-{}'.format(start_date, start_hour)
    end_time = '{}-{}'.format(end_date, end_hour)
    if data_source == 'biendata':
        url = 'https://biendata.com/competition/airquality/{}/{}/{}/{}'.format(city, start_time, end_time,
                                                                               DATA_API_TOKEN)
        response = requests.get(url)
        assert response.status_code == 200, 'api call failed'
        data = pd.read_csv(StringIO(response.text))
        data.drop(labels='id', axis=1, inplace=True)
        data.columns = ['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
        data.utc_time = pd.to_datetime(data.utc_time)
        start = pd.to_datetime(start_time + ':00:00')
        end = data.utc_time.max()
        if not partial_data:
            end = pd.to_datetime(end_time + ':00:00')
        data = normalize_datetime(data, keys=('stationId', 'utc_time'), min_time=start, max_time=end)
        data = data[['stationId', 'utc_time', 'PM2.5', 'PM10', 'O3']]
        if save and not os.path.exists('../input/{}-{}-{}.csv'.format(city, start_time, end_time)):
            data.to_csv('../input/{}-{}-{}.csv'.format(city, start_time, end_time), index=False)
        return data
    else:
        if city == 'bj':
            os.system('mkdir -p ../input/bj_pm_his')
            os.system('mkdir -p ../input/bj_o3_his')
            currentdate = pd.to_datetime(start_date)
            enddate = pd.to_datetime(end_date)
            enddate += pd.Timedelta(1, unit='d')
            bj_pm_data = []
            bj_o3_data = []
            while currentdate <= enddate:
                date_str = '{}{:02}{:02}'.format(currentdate.year, currentdate.month, currentdate.day)
                vprint(2, "# ---- get data for bj on date: {}".format(date_str))
                pm_url = 'http://beijingair.sinaapp.com/data/beijing/all/{}/csv'.format(date_str)
                o3_url = 'http://beijingair.sinaapp.com/data/beijing/extra/{}/csv'.format(date_str)
                pm_path = '../input/bj_pm_his/{}.csv'.format(date_str)
                o3_path = '../input/bj_o3_his/{}.csv'.format(date_str)
                if os.path.exists(pm_path):
                    data = pd.read_csv(pm_path)
                    bj_pm_data.append(data)
                else:
                    try:
                        response = requests.get(pm_url, timeout=3)
                        if response.status_code == 200:
                            data = pd.read_csv(StringIO(response.text))
                            data.columns = [col
                                            if col not in BJ_STATION_CHN_PINGYING_MAPPING
                                            else BJ_STATION_CHN_PINGYING_MAPPING[col]
                                            for col in data.columns]
                            if currentdate < enddate:
                                data.to_csv(pm_path, index=False)
                            bj_pm_data.append(data)
                        else:
                            vprint(2, 'response code is not 200')
                    except:
                        vprint(2, "# possibly timedout")
                if os.path.exists(o3_path):
                    data = pd.read_csv(o3_path)
                    bj_o3_data.append(data)
                else:
                    try:
                        response = requests.get(o3_url, timeout=3)
                        if response.status_code == 200:
                            data = pd.read_csv(StringIO(response.text))
                            data.columns = [col
                                            if col not in BJ_STATION_CHN_PINGYING_MAPPING
                                            else BJ_STATION_CHN_PINGYING_MAPPING[col]
                                            for col in data.columns]
                            if currentdate < enddate:
                                data.to_csv(o3_path, index=False)
                            bj_o3_data.append(data)
                        else:
                            vprint(2, 'response code is not 200')
                    except:
                        vprint(2, "# possibly timedout")
                time.sleep(1)  # be gentle to the server
                currentdate += pd.Timedelta(1, unit='D')
            aq_data = pd.concat(bj_pm_data + bj_o3_data, axis=0)
            if 'Unnamed: 0' in aq_data.columns:
                aq_data.drop('Unnamed: 0', axis=1, inplace=True)
            aq_data['utc_time'] = aq_data. \
                apply(lambda row: str(int(row['date'])) + ' {:02}:00:00'.\
                      format(row['hour']), axis=1)
            aq_data['utc_time'] = pd.to_datetime(aq_data['utc_time'])
            aq_data['utc_time'] = aq_data['utc_time'].map(lambda x: x - pd.Timedelta(8, unit='h'))
            aq_data = aq_data.loc[aq_data['type'].isin(['PM2.5', 'PM10', 'O3'])]
            aq_data.drop(labels=['date', 'hour'], axis=1, inplace=True)
            aq_data['utc_time'] = pd.to_datetime(aq_data['utc_time'])
            aq_data.drop_duplicates(subset=['type', 'utc_time'], keep='last', inplace=True)
            reshape_aq_data = []
            for station in BEIJING_STATIONS:
                s_aq_data = aq_data[[station] + ['utc_time', 'type']].copy()
                s_aq_data.columns = ['values', 'utc_time', 'type']
                d = s_aq_data.pivot(index='utc_time', columns='type', values='values').reset_index()
                d = pd.DataFrame(d.values, columns=d.columns.values)
                d['stationId'] = station
                d = d[['stationId', 'utc_time', 'PM2.5', 'PM10', 'O3']]
                reshape_aq_data.append(d)
            aq_data = pd.concat(reshape_aq_data, axis=0)
            aq_data['utc_time'] = pd.to_datetime(aq_data['utc_time'])
            aq_data.drop_duplicates(subset=['stationId', 'utc_time'], keep='last', inplace=True)
            aq_data = aq_data.loc[aq_data['utc_time'] >= pd.to_datetime(start_time)]
            if not partial_data:
                aq_data = aq_data.loc[aq_data['utc_time'] <= pd.to_datetime(end_time)]
            aq_data = normalize_datetime(aq_data)
            return aq_data
        else:
            os.system('mkdir -p ../input/ld_his')
            currentdate = pd.to_datetime(start_date)
            ld_aq_data = []
            end_date = str(pd.to_datetime(end_date) + pd.Timedelta(1, 'D')).split(' ')[0]
            start_y, start_m, start_d = start_date.split('-')
            end_y, end_m, end_d = end_date.split('-')
            start_m = MONTH_DIGIT_STR_MAPPING[start_m]
            end_m = MONTH_DIGIT_STR_MAPPING[end_m]
            start = start_d + start_m + start_y
            end = end_d + end_m + end_y
            for station in LONDON_STATIONS:
                url = 'http://api.erg.kcl.ac.uk/AirQuality/Data/Site/Wide/SiteCode={}'.format(station) + \
                      '/StartDate={}/EndDate={}/csv'.format(start, end)
                datapath = '../input/ld_his/{}_{}-{}.csv'.format(station, start, end)
                if os.path.exists(datapath):
                    data = pd.read_csv(datapath)
                    ld_aq_data.append(data)
                else:
                    vprint(2, '# ---- trying to get data from :' + url)
                    try:
                        response = requests.get(url, timeout=90)
                        if response.status_code == 200:
                            data = pd.read_csv(StringIO(response.text))
                            data.columns = pd.Series(data.columns).map(lambda x: x.split(': ')[-1])
                            data.columns = [col
                                            if col not in LONDON_API_COL_MAPPING
                                            else LONDON_API_COL_MAPPING[col]
                                            for col in data.columns]
                            data = data[['utc_time', 'PM2.5', 'PM10']]
                            data['stationId'] = station
                            ld_aq_data.append(data)
                            if currentdate < enddate:
                                data.to_csv(datapath)
                        else:
                            vprint(2, 'response code is not 200')
                    except:
                        vprint(2, "# possibly timedout")
            ld_aq_data = pd.concat(ld_aq_data, axis=0)
            if 'Unnamed: 0' in ld_aq_data.columns:
                ld_aq_data.drop('Unnamed: 0', axis=1, inplace=True)
            ld_aq_data = ld_aq_data[['stationId', 'utc_time', 'PM2.5', 'PM10']]
            ld_aq_data['O3'] = np.nan
            ld_aq_data['utc_time'] = pd.to_datetime(ld_aq_data['utc_time'])
            ld_aq_data.drop_duplicates(subset=['stationId', 'utc_time'], keep='last', inplace=True)
            ld_aq_data = ld_aq_data.loc[ld_aq_data['utc_time'] >= pd.to_datetime(start_time)]
            if not partial_data:
                ld_aq_data = ld_aq_data.loc[ld_aq_data['utc_time'] <= pd.to_datetime(end_time)]
            ld_aq_data = normalize_datetime(ld_aq_data)
            return ld_aq_data


def get_city_data(*, city: str, impute_with_lgbm: bool=False, partial_data: bool=False, vprint=print,
                  get_new_data: bool=False) -> pd.DataFrame:
    assert city in ['bj', 'ld'], 'invalid city'
    stations = BEIJING_STATIONS if city == 'bj' else LONDON_STATIONS
    end_date = get_date(pd.to_datetime(datetime.now()))
    vprint(1, '# ---- getting data for {}'.format(city))
    vprint(2, 'loading history data')
    his = pd.read_csv(filepath_or_buffer='../input/{}_api_his.csv'.format(city), parse_dates=['utc_time'])
    if get_new_data:
        vprint(2, 'loading new data')
        new = download_aq_data(city=city, start_date='2018-04-01', start_hour='00', end_date=end_date, end_hour='23',
                               save=False, partial_data=partial_data, data_source='alternative', vprint=vprint)
        data = pd.concat([his, new], axis=0)
    else:
        data = his
    data = data.loc[data['stationId'].isin(stations)][AQ_COL_NAMES]
    data = fix_nat(data)
    if impute_with_lgbm:
        data = lgbm_impute(data=data, city=city, vprint=vprint)
    return data


def evaluate(*, city: str, truth: pd.DataFrame, predictions: pd.DataFrame, measures=None) -> dict:
    scores = dict()
    stations = BEIJING_STATIONS if city == 'bj' else LONDON_STATIONS
    if not measures:
        measures = ['PM2.5', 'PM10', 'O3'] if city == 'bj' else ['PM2.5', 'PM10']
    merged = pd.merge(left=truth, right=predictions, how='left', on='test_id', suffixes=['_ans', '_pred'])
    for station in stations:
        for measure in measures:
            score = official_smape(
                merged.loc[merged['test_id'].map(lambda x: x.split('#')[0] == station)][measure + '_ans'],
                merged.loc[merged['test_id'].map(lambda x: x.split('#')[0] == station)][measure + '_pred']
            )
            scores['{}-{}'.format(station, measure)] = score
    for key in scores:
        if scores[key] == 2:
            scores[key] = np.nan
    return scores


def get_truth(*, city: str, data: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    truth = data.loc[(data['utc_time'] >= start_date) &
                     (data['utc_time'] < start_date + pd.Timedelta(value=2, unit='D'))].copy()  # type: pd.DataFrame
    truth['test_id'] = truth['stationId'] + ['#' + str(i) for i in range(48)] * int(truth.shape[0] / 48)
    if city == 'ld':
        truth['O3'] = 0
    truth.drop(labels=['stationId', 'utc_time'], axis=1, inplace=True)
    truth.dropna(inplace=True)
    return truth


def submit(subfile: str, description: str, filename: str):
    assert os.path.exists(subfile), 'submission file does not exist'
    files = {'files': open(subfile, 'rb')}
    data = {
        "user_id": USERNAME,
        "team_token": SUB_TOKEN,
        "description": description,
        "filename": filename,
    }
    url = 'https://biendata.com/competition/kdd_2018_submit/'
    response = requests.post(url, files=files, data=data)
    print(response.text)


