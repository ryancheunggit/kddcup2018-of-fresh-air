#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import os
import argparse
import pandas as pd
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from collections import namedtuple
from constants import BEIJING_STATIONS, LONDON_STATIONS, MEDIAN_WINDOWS, SUB_COLS
from util import (get_date, evaluate, get_verbose_print, fwbwmean_impute, get_city_data, str2bool, submit,
                  get_truth)


Setting = namedtuple('Setting', ['num', 'city', 'lgbm_impute', 'fwbw_mean_impute', 'n_thread', 'method',  'windows'])


def get_summary(window_sizes: list, data: pd.DataFrame, impute_methods: list, predict_start: pd.Timestamp,
                verbose_level: int=500, method='median'):
    if verbose_level >= 2:
        print('medians from window of size {}'.format(window_sizes))
    window_1 = data.loc[(data.utc_time >= predict_start - pd.Timedelta(window_sizes + 1, unit='D')) &
                        (data.utc_time < predict_start - pd.Timedelta(2, unit='D'))].copy()
    window_2 = data.loc[(data.utc_time >= predict_start - pd.Timedelta(window_sizes, unit='D')) &
                        (data.utc_time < predict_start - pd.Timedelta(1, unit='D'))].copy()
    if impute_methods:
        window_1 = fwbwmean_impute(data=window_1, main_method=impute_methods[0], residual_method=impute_methods[1])
        window_2 = fwbwmean_impute(data=window_2, main_method=impute_methods[0], residual_method=impute_methods[1])
    window_1.loc[:, 'hour'] = window_1['utc_time'].map(lambda x: x.hour)
    summary_1 = window_1. \
        groupby(['stationId', 'hour']). \
        agg({'PM2.5': method, 'PM10': method, 'O3': method}). \
        reset_index()
    summary_1.sort_values(by=['stationId', 'hour'], axis=0, inplace=True)
    summary_1['test_id'] = ["#".join([k, str(v)]) for k, v in zip(summary_1['stationId'], 48 * list(range(0, 24)))]

    window_2.loc[:, 'hour'] = window_2['utc_time'].map(lambda x: x.hour)
    summary_2 = window_2. \
        groupby(['stationId', 'hour']). \
        agg({'PM2.5': method, 'PM10': method, 'O3': method}). \
        reset_index()
    summary_2.sort_values(by=['stationId', 'hour'], axis=0, inplace=True)
    summary_2['test_id'] = ["#".join([k, str(v)]) for k, v in zip(summary_1['stationId'], 48 * list(range(24, 48)))]
    medians = pd.concat([summary_1, summary_2], axis=0)[SUB_COLS]
    return medians


def rolling_summary(*, sub: pd.DataFrame, data: pd.DataFrame, predict_start: pd.Timestamp, windows: list,
                    method='median', impute_methods: list, n_thread: int=1, vprint=print):
    vprint(2, 'rolling median prediction starting {}'.format(get_date(predict_start)))
    vprint(2, data.utc_time.min())
    vprint(2, data.utc_time.max())
    verbose_level = getattr(vprint, 'verbose_level', 500)
    get_medians_thread = partial(get_summary, data=data, predict_start=predict_start, method=method,
                                 impute_methods=impute_methods, verbose_level=verbose_level)
    pool = Pool(n_thread)
    medians = pool.map(get_medians_thread, windows)
    pool.terminate()
    pool.close()
    predictions = medians[0].copy()
    for col in ['PM2.5', 'PM10', 'O3']:
        predictions[col] = pd.concat([median[col] for median in medians], axis=1).median(axis=1)
    submissions = pd.merge(left=sub[['test_id']], right=predictions[SUB_COLS], how='left')
    return submissions


def fit_predict_score(*, tr: pd.DataFrame, ts: pd.DataFrame, sub: pd.DataFrame, city: str, windows: list,
                      start_date: pd.Timestamp, end_date: pd.Timestamp, impute_methods: list, n_thread: int,
                      method: str='median', vprint=print) -> pd.DataFrame:
    current_date = start_date
    scores_df = []
    while current_date < end_date:
        vprint(1, '# --- fitting predicting evaluating on {}'.format(get_date(current_date)))
        predictions = rolling_summary(sub=sub, data=tr, predict_start=current_date, windows=windows, n_thread=n_thread,
                                      method=method, impute_methods=impute_methods, vprint=vprint)
        truth = get_truth(city=city, data=ts, start_date=current_date + pd.Timedelta(1, unit='D'))
        scores = evaluate(city=city, truth=truth, predictions=predictions)
        scores['smape'] = pd.Series(scores).mean()
        scores['date'] = get_date(current_date)
        vprint(1, scores['smape'])
        current_date += pd.Timedelta(value=1, unit='D')
        scores_df.append(scores)
    scores_df = pd.DataFrame(scores_df)
    scores_df = scores_df[['date', 'smape'] + [col for col in scores_df.columns if col not in ['date', 'smape']]]
    return scores_df


def run_setting(*, setting: Setting, start_date: pd.Timestamp, end_date: pd.Timestamp, vprint=print) -> pd.DataFrame:
    sub = pd.read_csv('../input/sample_submission.csv')
    get_new_data = end_date > pd.to_datetime('2018-03-28')
    test_data = get_city_data(city=setting.city, vprint=vprint, impute_with_lgbm=False, get_new_data=get_new_data)
    if setting.lgbm_impute:
        train_data = get_city_data(city=setting.city, vprint=vprint, impute_with_lgbm=setting.lgbm_impute,
                                   get_new_data=get_new_data)
    else:
        train_data = test_data.copy()
    impute_methods = ['fwbw_day', 'mean'] if setting.fwbw_mean_impute else []
    scores_df = fit_predict_score(tr=train_data, ts=test_data, sub=sub, city=setting.city, windows=setting.windows,
                                  start_date=start_date, end_date=end_date, impute_methods=impute_methods,
                                  method=setting.method, n_thread=setting.n_thread, vprint=vprint)
    outfile_name = 'rolling_summary_experiment_{}_{}.csv'.format(setting.num, setting.city)
    outfile_path = '../summaries/{}'.format(outfile_name)
    if os.path.exists(outfile_path):
        df = pd.read_csv(outfile_path)
        scores_df = pd.concat([scores_df, df], axis=0)
        scores_df.drop_duplicates(inplace=True)
        scores_df.sort_values(by='date', inplace=True)
    scores_df.to_csv('../summaries/{}'.format(outfile_name), index=False)
    with open('../summaries/settings.txt', 'a') as f:
        f.write(str(setting) + ' Summary File: ' + outfile_name)
    vprint(1, '# ---- mean {} '.format(scores_df['smape'].mean()))
    return scores_df


def experiment(settings_to_run: list, start_date: str, end_date: str, n_thread: int, verbose: int=2):
    vprint = get_verbose_print(verbose_level=verbose)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    model_num = 0
    settings = []
    for city in ['bj', 'ld']:
        for lgbm_impute in [True, False]:
            for fwbw_mean_impute in [True, False]:
                for method in ['median', 'mean']:
                    for window_type in ['golden_8', 'golden_7', 'golden_6', 'golden_5', 'golden_4', 'fib_9', 'fib_8',
                                        'fib_7', 'fib_6', 'fib_5']:
                        model_num += 1
                        settings.append(Setting(num=model_num, city=city, lgbm_impute=lgbm_impute,
                                                fwbw_mean_impute=fwbw_mean_impute, n_thread=n_thread, method=method,
                                                windows=MEDIAN_WINDOWS[window_type]))

    for idx, setting in enumerate(settings):
        if (idx + 1) in settings_to_run:
            vprint(0, '# ==== running setting {}'.format(setting.num))
            vprint(1, '# ---- setting is {}'.format(str(setting)))
            _ = run_setting(setting=setting, start_date=start_date, end_date=end_date, vprint=vprint)


def predict(*, pred_date: str, bj_windows: str='golden_8', ld_windows: str='golden_8', bj_method: str='median',
            ld_method: str='median', bj_lgbm: bool=True, ld_lgbm: bool=True, bj_fwbw: bool=True, ld_fwbw: bool=True,
            n_thread: int=8, save: bool=True, dosubmit: bool=False, suffix: str='dummy', verbose: int=2):
    vprint = get_verbose_print(verbose_level=verbose)
    pred_date = pd.to_datetime(pred_date)
    get_new_data = pred_date > pd.to_datetime('2018-03-28')
    sub = pd.read_csv("../input/sample_submission.csv")
    OUTDIR = '../submission/sub_{}-{}-{}'.format(pred_date.year, pred_date.month, pred_date.day)
    os.system('mkdir -p {}'.format(OUTDIR))
    predict_start_day = pred_date + pd.Timedelta(1, unit='D')
    predict_start = pd.to_datetime(get_date(predict_start_day))
    bj_data = get_city_data(city='bj', vprint=vprint, impute_with_lgbm=bj_lgbm, get_new_data=get_new_data)
    ld_data = get_city_data(city='ld', vprint=vprint, impute_with_lgbm=ld_lgbm, get_new_data=get_new_data)
    vprint(2, bj_data.head())
    vprint(2, bj_data.loc[bj_data['stationId']!= 'zhiwuyuan_aq'].tail())
    vprint(2, ld_data.head())
    vprint(2, ld_data.tail())
    bj_fwbw_impute_methods = ['day', 'mean'] if bj_fwbw else []
    ld_fwbw_impute_methods = ['day', 'mean'] if ld_fwbw else []
    bj_pred = rolling_summary(sub=sub, data=bj_data, predict_start=predict_start, windows=MEDIAN_WINDOWS[bj_windows],
                              n_thread=n_thread, method=bj_method, impute_methods=bj_fwbw_impute_methods, vprint=vprint)
    ld_pred = rolling_summary(sub=sub, data=ld_data, predict_start=predict_start, windows=MEDIAN_WINDOWS[ld_windows],
                              n_thread=n_thread, method=ld_method, impute_methods=ld_fwbw_impute_methods, vprint=vprint)
    submissions = sub.copy()
    bj_cond = submissions['test_id'].map(lambda x: x.split('#')[0] in BEIJING_STATIONS)
    ld_cond = submissions['test_id'].map(lambda x: x.split('#')[0] in LONDON_STATIONS)
    submissions.loc[bj_cond] = bj_pred.loc[bj_cond].values
    submissions.loc[ld_cond] = ld_pred.loc[ld_cond].values
    submissions['PM2.5'] = submissions['PM2.5'].map(lambda x: max(0, x))
    submissions['PM10'] = submissions['PM10'].map(lambda x: max(0, x))
    submissions['O3'] = submissions['O3'].map(lambda x: max(0, x))

    if save:
        if not suffix:
            filepath = '{}/model_{}_sub.csv'.format(OUTDIR, 3)
        else:
            filepath = '{}/model_{}_sub_{}.csv'.format(OUTDIR, 3, suffix)
        submissions.to_csv(filepath, index=False)

        if dosubmit:
            submit(subfile=filepath,
                   description='model_{}_{}'.format(3, str(predict_start).split()[0]),
                   filename='model_{}_sub_{}.csv'.format(3, str(predict_start).split()[0])
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment or predicting with rolling summaries model')
    parser.add_argument('--mode', type=str, default='exp', help='whether to do experiment or make prediction')
    parser.add_argument('--pred_date', type=str, default=None, help='to generate prediction for this date')
    parser.add_argument('--pred_bj_windows', type=str, default='golden_8', help='what windows to use for bj')
    parser.add_argument('--pred_ld_windows', type=str, default='golden_8', help='what windows to use for ld')
    parser.add_argument('--pred_bj_method', type=str, default='median', help='what method to use for bj')
    parser.add_argument('--pred_ld_method', type=str, default='median', help='what method to use for ld')
    parser.add_argument('--pred_bj_lgbm', type=str2bool, default='True', help='whether to impute with lgbm for bj')
    parser.add_argument('--pred_ld_lgbm', type=str2bool, default='True', help='whether to impute with lgbm for ld')
    parser.add_argument('--pred_bj_fwbw', type=str2bool, default='True', help='whether to impute with fwbw for bj')
    parser.add_argument('--pred_ld_fwbw', type=str2bool, default='True', help='whether to impute with fwbw for ld')
    parser.add_argument('--save', type=str2bool, default='True', help='whether to save submission file')
    parser.add_argument('--save_suffix', type=str, default='alt_lgb_roll', help='suffix append to submission filename')
    parser.add_argument('--submit', type=str2bool, default='False', help='whether to submit submission file')
    parser.add_argument('--exp_start_date', type=str, default=None, help='date to start experiment, inclusive')
    parser.add_argument('--exp_end_date', type=str, default=None, help='date to end experiment, exclusive')
    parser.add_argument('--n_thread', type=int, default=8, help='number of threads to run experiment')
    parser.add_argument('--verbose', type=int, default=2, help='verbose level')
    args = parser.parse_args()
    if args.mode == 'exp':
        experiment(list(range(1, 81)), args.exp_start_date, args.exp_end_date, args.n_thread, args.verbose)
    else:
        if not args.pred_date
            args.pred_date = get_date(pd.to_datetime(datetime.now()))
        predict(pred_date=args.pred_date, bj_windows=args.pred_bj_windows, ld_windows=args.pred_ld_windows,
                bj_method=args.pred_bj_method, ld_method=args.pred_ld_method, bj_lgbm=args.pred_bj_lgbm,
                ld_lgbm=args.pred_ld_lgbm, bj_fwbw=args.pred_bj_fwbw, ld_fwbw=args.pred_ld_fwbw, n_thread=args.n_thread,
                save=args.save, dosubmit=args.submit, suffix=args.save_suffix, verbose=int(args.verbose))