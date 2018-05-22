#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
from constants import BEIJING_STATIONS, LONDON_STATIONS, AQ_COL_NAMES
from sklearn.model_selection import GridSearchCV
from util import download_aq_data, fix_nat, get_date, lgbm_impute
from datetime import datetime
import pickle


def lgb_cv(df, stationId, measure):
    stations = df['stationId'].unique()
    data = [df.loc[df['stationId'] == station][measure].values for station in stations]
    data = pd.DataFrame(np.array(data)).T
    data.columns = stations
    y = data.loc[~data[stationId].isnull()][stationId]
    X = data.loc[~data[stationId].isnull()][[col for col in stations if col != stationId]]
    lgb_model = lgb.LGBMRegressor(objective='regression', n_jobs=1)
    params_dist = {
        'learning_rate': [0.05, 0.1, 0.2],
        'num_leaves': [27, 31, 35],
        'n_estimators': [80, 100],
        'subsample': [0.3, .5, .7],
        'colsample_bytree': [0.3, .5, .7],
        'reg_alpha': [0, .2],
        'reg_lambda': [0, .2]
    }
    results = GridSearchCV(
        estimator=lgb_model,
        param_grid=params_dist,
        scoring='neg_mean_absolute_error',
        n_jobs=8,
        cv=5,
        refit=True,
        verbose=1,
        return_train_score=True
    ).fit(X, y)
    print(results.best_params_)
    return results.best_estimator_


def main(mode: str='train'):
    if mode == 'train':
        os.system('mkdir -p ../models')
        assert os.path.exists('../input/bj_api_his.csv'), 'run download_data.sh to get data first'
        assert os.path.exists('../input/ld_api_his.csv'), 'run download_data.sh to get data first'
        bj_his = pd.read_csv(filepath_or_buffer='../input/bj_api_his.csv', parse_dates=['utc_time'])
        ld_his = pd.read_csv(filepath_or_buffer='../input/ld_api_his.csv', parse_dates=['utc_time'])
        bj_lgb_models = {
            '{}-{}'.format(stationId, measure): lgb_cv(bj_his, stationId, measure)
            for stationId in BEIJING_STATIONS
            for measure in ['PM2.5', 'PM10', 'O3']
        }
        with open("../models/bj_lgbm.pkl", 'wb') as f:
            pickle.dump(bj_lgb_models, f)
        ld_lgb_models = {
            '{}-{}'.format(stationId, measure): lgb_cv(ld_his, stationId, measure)
            for stationId in LONDON_STATIONS
            for measure in ['PM2.5', 'PM10']
        }
        with open("../models/ld_lgbm.pkl", 'wb') as f:
            pickle.dump(ld_lgb_models, f)
        print('# ---- DONE ---- #')

    if mode == 'impute':
        assert os.path.exists('../models/bj_lgbm.pkl'), 'model not trained yet'
        assert os.path.exists('../models/ld_lgbm.pkl'), 'model not trained yet'
        bj_his = pd.read_csv(filepath_or_buffer='../input/bj_api_his.csv', parse_dates=['utc_time'])
        ld_his = pd.read_csv(filepath_or_buffer='../input/ld_api_his.csv', parse_dates=['utc_time'])
        end_date = get_date(pd.to_datetime(datetime.now()) + pd.Timedelta(1, 'D'))
        bj_new = download_aq_data(
            city='bj',
            start_date='2018-04-01',
            start_hour='00',
            end_date=end_date,
            end_hour='23',
            save=False,
            partial_data=False,
            data_source='alternative'
        )
        ld_new = download_aq_data(
            city='ld',
            start_date='2018-04-01',
            start_hour='00',
            end_date=end_date,
            end_hour='23',
            save=False,
            partial_data=False,
            data_source='alternative'
        )
        bj_new = bj_new.loc[bj_new.utc_time < pd.to_datetime(today) - pd.Timedelta(1, 'D')]
        ld_new = ld_new.loc[ld_new.utc_time < pd.to_datetime(today) - pd.Timedelta(1, 'D')]
        bj_data = pd.concat([bj_his, bj_new], axis=0)
        ld_data = pd.concat([ld_his, ld_new], axis=0)
        ld_data = ld_data.loc[ld_data.stationId.isin(LONDON_STATIONS)]
        bj_data = bj_data.loc[bj_data.stationId.isin(BEIJING_STATIONS)]
        bj_data = bj_data[AQ_COL_NAMES]
        ld_data = ld_data[AQ_COL_NAMES]
        bj_data = fix_nat(bj_data)
        ld_data = fix_nat(ld_data)

        bj_data = lgbm_impute(data=bj_data, city='bj')
        ld_data = lgbm_impute(data=ld_data, city='ld')
        data = pd.concat([bj_data, ld_data], axis=0)
        data = fix_nat(data)
        data.to_csv('../input/lgb_imputed_new_source_2014-03-31-_{}.csv'.format(today), index=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        mode = 'train'
    else:
        mode = str(sys.argv[1])
    assert mode in ['train', 'impute'], 'invalid mode'
    main(mode)
