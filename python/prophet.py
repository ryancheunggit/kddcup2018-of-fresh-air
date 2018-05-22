#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import argparse
import os
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from functools import partial
from collections import namedtuple
from constants import BEIJING_STATIONS, LONDON_STATIONS, SUB_COLS
from util import (get_verbose_print, get_truth, get_date, evaluate, str2bool, submit, suppress_stdout_stderr,
                  get_city_data)
from fbprophet import Prophet
Setting = namedtuple('Setting', ['num', 'city', 'n_thread', 'history_length', "changepoint_scale", "num_changepoints"])


def fb_fit_predict(station_measure, df, scale, changepoints, future, verbose_level: int=500):
    station, measure = station_measure
    if verbose_level >= 2:
        print('fitting model on {} {}'.format(station, measure))
    model = Prophet(changepoint_prior_scale=scale, n_changepoints=changepoints)
    model_data = df.loc[df.stationId == station][['utc_time', measure]].copy()
    model_data.columns = ['ds', 'y']
    with suppress_stdout_stderr():
        model.fit(model_data)
    forecast = model.predict(future)
    return [station, measure, forecast['yhat'].values]


def fbprophet(sub, data, current_date, history_length, changepoint_scale, num_changepoints, n_thread=8, vprint=print):
    submission = sub.copy()
    submission['stationId'] = submission['test_id'].map(lambda x: x.split('#')[0])
    future = pd.DataFrame({'ds': [current_date + pd.Timedelta(i, unit='h') for i in range(48)]})
    vprint(2, 'future starts {}'.format(future.ds.min()))
    vprint(2, 'future ends {}'.format(future.ds.max()))
    df = data.loc[data['utc_time'] <= current_date - pd.Timedelta(9, unit='h')].copy()
    df = df.loc[df['utc_time'] >= df['utc_time'].max() - pd.Timedelta(history_length, unit='D')]
    vprint(2, 'fbprophet prediction on date {}'.format(get_date(current_date)))
    vprint(2, df.utc_time.min())
    vprint(2, df.utc_time.max())
    verbose_level = getattr(vprint, 'verbose_level', 500)
    stations = data.stationId.unique().tolist()
    run_list = []
    for station in stations:
        measures = ['PM2.5', 'PM10', 'O3']
        if station in LONDON_STATIONS:
            measures.remove('O3')
        for measure in measures:
            run_list.append((station, measure))
    fb_fit_predict_thread = partial(fb_fit_predict, df=df, scale=changepoint_scale, changepoints=num_changepoints,
                                    future=future, verbose_level=verbose_level)
    pool = Pool(n_thread)
    results = pool.map(fb_fit_predict_thread, run_list)
    pool.terminate()
    pool.close()
    for result in results:
        station, measure, values = result
        submission.loc[submission['stationId'] == station, measure] = values
    return submission[SUB_COLS]


def fit_predict_score(sub, tr, ts, start_date, end_date, city, history_length, changepoint_scale, num_changepoints,
                      n_thread, vprint=print) -> pd.DataFrame:
    current_date = start_date
    scores_df = []
    while current_date < end_date:
        vprint(1, '# --- fitting predicting evaluating on {}'.format(get_date(current_date)))
        predictions = fbprophet(sub=sub, data=tr, current_date=current_date, history_length=history_length,
                                changepoint_scale=changepoint_scale, num_changepoints=num_changepoints,
                                n_thread=n_thread, vprint=vprint)
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
    train_data = get_city_data(city=setting.city, vprint=vprint, impute_with_lgbm=True, get_new_data=get_new_data)
    test_data = get_city_data(city=setting.city, vprint=vprint, impute_with_lgbm=False, get_new_data=get_new_data)
    scores_df = fit_predict_score(
        sub=sub, tr=train_data, ts=test_data, start_date=start_date, end_date=end_date, city=setting.city,
        history_length=setting.history_length, changepoint_scale=setting.changepoint_scale,
        num_changepoints=setting.num_changepoints, n_thread=setting.n_thread, vprint=vprint)
    outfile_name = 'fbprophet_experiment_{}_{}.csv'.format(setting.num, setting.city)
    outfile_path = '../summaries/{}'.format(outfile_name)
    if os.path.exists(outfile_path):
        df = pd.read_csv(outfile_path)
        scores_df = pd.concat([scores_df, df], axis=0)
        scores_df.drop_duplicates(inplace=True)
        scores_df.sort_values(by='date', inplace=True)
    scores_df.to_csv('../summaries/{}'.format(outfile_name), index=False)
    with open('../summaries/fb_settings.txt', 'a') as f:
        f.write(str(setting) + ' Summary File: ' + outfile_name)
    vprint(1, '# ---- mean {} '.format(scores_df['smape'].mean()))
    return scores_df


def experiment(settings_to_run: list, start_date: str, end_date: str, n_thread: int=8, verbose: int=2):
    vprint = get_verbose_print(verbose_level=verbose)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    model_num = 0
    settings = []
    for city in ['bj', 'ld']:
        for history_length in [475, 294, 181, 112]:
            for changepoint_scale in [0.005, 0.05, 0.5]:
                model_num += 1
                settings.append(Setting(num=model_num, city=city, n_thread=n_thread, history_length=history_length,
                                        changepoint_scale=changepoint_scale, num_changepoints=25))

    for city in ['bj', 'ld']:
        for changepoint_scale in [0.001, 0.003, 0.007, 0.01]:
            for num_changepoint in [13, 25, 49, 67, 99]:
                model_num += 1
                settings.append(Setting(num=model_num, city=city, n_thread=n_thread, history_length=475,
                                        changepoint_scale=changepoint_scale, num_changepoints=num_changepoint))

    for city in ['bj', 'ld']:
        for changepoint_scale in [0.005]:
            for num_changepoint in [13, 49, 67, 99]:
                model_num += 1
                settings.append(Setting(num=model_num, city=city, n_thread=n_thread, history_length=475,
                                        changepoint_scale=changepoint_scale, num_changepoints=num_changepoint))

    for idx, setting in enumerate(settings):
        if (idx + 1) in settings_to_run:
            vprint(0, '# === running setting {}'.format(setting.num))
            vprint(1, '# ---- setting is {}'.format(str(setting)))
            _ = run_setting(setting=setting, start_date=start_date, end_date=end_date, vprint=vprint)


def predict(*, pred_date: str, bj_his_length: int, ld_his_length: int, bj_npoints: int, ld_npoints: int, bj_scale:float,
            ld_scale: float, n_thread: int=8, save: bool=True, dosubmit: bool=False, suffix: str='dummy',
            verbose: int=2):
    vprint = get_verbose_print(verbose_level=verbose)
    pred_date = pd.to_datetime(pred_date)
    get_new_data = pred_date > pd.to_datetime('2018-03-28')
    sub = pd.read_csv("../input/sample_submission.csv")
    OUTDIR = '../submission/sub_{}-{}-{}'.format(pred_date.year, pred_date.month, pred_date.day)
    os.system('mkdir -p {}'.format(OUTDIR))
    predict_start_day = pred_date + pd.Timedelta(1, unit='D')
    predict_start = pd.to_datetime(get_date(predict_start_day))
    bj_data = get_city_data(city='bj', vprint=vprint, impute_with_lgbm=True, partial_data=True, get_new_data=get_new_data)
    ld_data = get_city_data(city='ld', vprint=vprint, impute_with_lgbm=True, partial_data=True, get_new_data=get_new_data)
    vprint(2, bj_data.head())
    vprint(2, bj_data.loc[bj_data['stationId']!= 'zhiwuyuan_aq'].tail())
    vprint(2, ld_data.head())
    vprint(2, ld_data.tail())
    bj_pred = fbprophet(sub=sub, data=bj_data, current_date=predict_start, history_length=bj_his_length,
                        changepoint_scale=bj_scale, num_changepoints=bj_npoints, n_thread=n_thread, vprint=vprint)
    ld_pred = fbprophet(sub=sub, data=ld_data, current_date=predict_start, history_length=ld_his_length,
                        changepoint_scale=ld_scale, num_changepoints=ld_npoints, n_thread=n_thread, vprint=vprint)
    submissions = sub.copy()
    bj_cond = submissions['test_id'].map(lambda x: x.split('#')[0] in BEIJING_STATIONS)
    ld_cond = submissions['test_id'].map(lambda x: x.split('#')[0] in LONDON_STATIONS)
    submissions.loc[bj_cond, ['PM2.5', 'PM10', 'O3']] = bj_pred.loc[bj_cond,  ['PM2.5', 'PM10', 'O3']].values
    submissions.loc[ld_cond, ['PM2.5', 'PM10']] = ld_pred.loc[ld_cond, ['PM2.5', 'PM10']].values
    submissions['PM2.5'] = submissions['PM2.5'].map(lambda x: max(0, x))
    submissions['PM10'] = submissions['PM10'].map(lambda x: max(0, x))
    submissions['O3'] = submissions['O3'].map(lambda x: max(0, x))
    submissions = submissions[['test_id', 'PM2.5', 'PM10', 'O3']]
    if save:
        if not suffix:
            filepath = '{}/model_{}_sub.csv'.format(OUTDIR, 4)
        else:
            filepath = '{}/model_{}_sub_{}.csv'.format(OUTDIR, 4, suffix)
        submissions.to_csv(filepath, index=False)

        if dosubmit:
            submit(subfile=filepath,
                   description='model_{}_{}'.format(4, str(predict_start).split()[0]),
                   filename='model_{}_sub_{}.csv'.format(4, str(predict_start).split()[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment or predicting with prophet model')
    parser.add_argument('--mode', type=str, default='exp', help='whether to do experiment or make prediction')
    parser.add_argument('--pred_date', type=str, default=None, help='to generate prediction for this date')
    parser.add_argument('--pred_bj_his_length', type=int, default=475, help='number of days of history to use for bj')
    parser.add_argument('--pred_ld_his_length', type=int, default=475, help='number of days of history to use for ld')
    parser.add_argument('--pred_bj_scale', type=float, default=0.003, help='regularization param bj')
    parser.add_argument('--pred_ld_scale', type=float, default=0.001, help='regularization param ld')
    parser.add_argument('--pred_bj_npoints', type=int, default=67, help='complexity param bj')
    parser.add_argument('--pred_ld_npoints', type=int, default=67, help='complexity param ld')
    parser.add_argument('--save', type=str2bool, default='True', help='whether to save submission file')
    parser.add_argument('--save_suffix', type=str, default='alt_lgb_fin', help='suffix append to submission filename')
    parser.add_argument('--submit', type=str2bool, default='False', help='whether to submit submission file')
    parser.add_argument('--exp_start_date', type=str, default='2018-03-21', help='date to start experiment, inclusive')
    parser.add_argument('--exp_end_date', type=str, default='2018-05-08', help='date to end experiment, exclusive')
    parser.add_argument('--n_thread', type=int, default=8, help='number of threads to run experiment')
    parser.add_argument('--verbose', type=int, default=2, help='verbose level')
    args = parser.parse_args()
    if args.mode == 'exp':
        experiment([1,2,3], args.exp_start_date, args.exp_end_date, args.n_thread, args.verbose)
    else:
        if not args.pred_date
            args.pred_date = get_date(pd.to_datetime(datetime.now()))
        predict(pred_date=args.pred_date, bj_his_length=args.pred_bj_his_length, ld_his_length=args.pred_ld_his_length,
                bj_scale=args.pred_bj_scale, ld_scale=args.pred_ld_scale, bj_npoints=args.pred_bj_npoints,
                ld_npoints=args.pred_ld_npoints, n_thread=args.n_thread, save=args.save, dosubmit=args.submit,
                suffix=args.save_suffix, verbose=args.verbose)
