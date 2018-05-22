#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import namedtuple
from constants import BEIJING_STATIONS, LONDON_STATIONS, SUB_COLS, MEDIAN_WINDOWS
from util import (get_verbose_print, get_truth, get_date, evaluate, str2bool, submit, suppress_stdout_stderr,
                  get_city_data, impute, long_to_wide, wide_make_fw_x_y)
import keras.backend as k
from keras.layers import (Input, Concatenate, BatchNormalization, Dropout, Dense)
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupKFold

Setting = namedtuple('Setting', ['num', 'city', 'history_length', 'num_shifts', 'median_shifts', 'median_windows',
                                 'dropout_rate', 'l2_strength', 'units', 'batch_size'])


def smape_loss(y_true, y_pred):
    return 2 * k.mean(k.abs(y_pred - y_true) / (k.abs(y_pred) + k.abs(y_true)), axis=-1)


def get_keras_model(input_dim: int, dropout_rate: float=0.5, l2_strength: float=0.0001, units: list=(128, 128, 64, 64),
                    lr: float=0.005):
    inp = Input(shape=(input_dim,), dtype='float32')
    x = Dense(units=units[0], activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_strength))(inp)
    x = Dropout(dropout_rate)(x)
    x = Concatenate()([inp, x])
    x = Dense(units=units[1], activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_strength))(x)
    x = BatchNormalization(beta_regularizer=l2(l2_strength), gamma_regularizer=l2(l2_strength))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=units[2], activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_strength))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=units[3], activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_strength))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=48, activation='linear', kernel_initializer='lecun_uniform', kernel_regularizer=l2(l2_strength))(x)
    model = Model(inputs=inp, outputs=x)
    optimizer = Adam(lr=lr, clipvalue=1, clipnorm=1)
    loss = smape_loss
    model.compile(loss=loss, optimizer=optimizer)
    return model


def run_setting(setting, verbose_level=2, start_date, end_date, n_folds=5, skip=1):
    num, city, history_length, num_shifts, median_shifts, median_windows, dropout_rate, l2_strength, units, batch_size = setting
    median_windows = MEDIAN_WINDOWS[median_windows]
    vprint = get_verbose_print(verbose_level)
    sub = pd.read_csv('../input/sample_submission.csv')
    get_new_data = end_date > pd.to_datetime('2018-03-28')
    test_data = get_city_data(city=city, vprint=vprint, impute_with_lgbm=False, get_new_data=get_new_data)
    train_data = test_data.copy()  # type: pd.DataFrame
    vprint(2, train_data.head())
    vprint(2, train_data.loc[train_data['stationId'] != 'zhiwuyuan_aq'].tail())
    train_data = impute(train_data, lgbm=True, hour=True, mean=True)
    vprint(2, train_data.head())
    vprint(2, train_data.loc[train_data['stationId'] != 'zhiwuyuan_aq'].tail())
    w_train_data = long_to_wide(train_data)
    current_date = start_date
    scores_df = []
    STATIONS = BEIJING_STATIONS if city == 'bj' else LONDON_STATIONS
    while current_date < end_date:
        vprint(1, "running experiment for {} at {}".format(current_date, datetime.now()))

        train_split_date = current_date - pd.Timedelta(3, unit='D')

        x_train, y_train = wide_make_fw_x_y(
            wdata=w_train_data, ldata=train_data, split_date=train_split_date, history_length=history_length,
            num_shifts=num_shifts, use_medians=True, median_shifts=median_shifts, window_sizes=median_windows,
            for_prediction=False, n_thread=8, vprint=vprint, save_feature=True, use_cache=True,
            window_name=setting.median_windows)

        x_test, _ = wide_make_fw_x_y(
            wdata=w_train_data, ldata=train_data, split_date=current_date, history_length=history_length,
            num_shifts=1, use_medians=True, median_shifts=median_shifts, window_sizes=median_windows,
            for_prediction=False, n_thread=8, vprint=vprint, save_feature=True, use_cache=True,
            window_name=setting.median_windows)

        x_train = x_train.loc[x_train['stationId'].map(lambda x: x.split('#')[0] in [s for s in STATIONS if s != 'zhiwuyuan_aq'])]
        y_train = y_train.loc[y_train['stationId'].map(lambda x: x.split('#')[0] in [s for s in STATIONS if s != 'zhiwuyuan_aq'])]
        x_test = x_test.loc[x_test['stationId'].map(lambda x: x.split('#')[0] in STATIONS)]

        subs = []
        min_valid_smape = []

        groups = x_train['stationId'].map(lambda x: x.split('#')[0])
        group_kfold = GroupKFold(n_splits=n_folds)
        splits = list(group_kfold.split(X=x_train, groups=groups))

        for it, (train_idx, val_idx) in enumerate(splits):
            vprint(2, '# ---- fold {} ----'.format(it + 1))
            model = get_keras_model(input_dim=x_train.shape[1] - 1, dropout_rate=dropout_rate, l2_strength=l2_strength, units=units)
            if it == 0:
                vprint(1, model.summary())
            history = model.fit(
                x=x_train.iloc[train_idx, 1:].values,
                y=y_train.iloc[train_idx, 1:].values,
                validation_data=(x_train.iloc[val_idx, 1:].values, y_train.iloc[val_idx, 1:].values),
                batch_size=batch_size,
                epochs=65535,
                verbose=0,
                callbacks=[
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=verbose_level),
                    EarlyStopping(monitor='val_loss', patience=30, verbose=verbose_level),
                    ModelCheckpoint(filepath='./model_checkpoint_{}.hdf5'.format(num), monitor='val_loss',
                                    save_best_only=True, save_weights_only=True, mode='min')
                ])
            min_valid_smape.append(np.min(history.history['val_loss']))
            predictions = model.predict(x_test.iloc[:, 1:], verbose=verbose_level)
            predictions = pd.DataFrame(predictions)
            predictions['stationId'] = x_test['stationId'].map(lambda x: x.split('#')[0]).tolist()
            predictions['measure'] = x_test['stationId'].map(lambda x: x.split('#')[1]).tolist()
            vprint(2, '# ---- formatting submission df ----')
            for idx, row in tqdm(predictions.iterrows()):
                values = row[:48].values
                sub.loc[sub.test_id.isin([row['stationId'] + '#' + str(i) for i in range(48)]), row['measure']] = values
            subs.append(sub[SUB_COLS])
        vprint(2, 'mean {}, std {}'.format(np.mean(min_valid_smape), np.std(min_valid_smape)))
        submissions = subs[0]
        for sub in subs[1:]:
            submissions[['PM2.5', 'PM10', 'O3']] += sub[['PM2.5', 'PM10', 'O3']]
        submissions[['PM2.5', 'PM10', 'O3']] /= n_folds

        truth = get_truth(city=city, data=test_data, start_date=current_date + pd.Timedelta(1, unit='D'))
        scores = evaluate(city=city, truth=truth, predictions=submissions)
        if 'zhiwuyuan_aq-O3' in scores:
            scores['zhiwuyuan_aq-O3'] = np.nan
        if 'zhiwuyuan_aq-PM2.5' in scores:
            scores['zhiwuyuan_aq-PM2.5'] = np.nan
        if 'zhiwuyuan_aq-PM10' in scores:
            scores['zhiwuyuan_aq-PM10'] = np.nan
        scores['smape'] = pd.Series(scores).mean()
        scores['date'] = get_date(current_date)
        vprint(1, scores['smape'])
        current_date += pd.Timedelta(value=skip, unit='D')
        scores_df.append(scores)
    scores_df = pd.DataFrame(scores_df)
    scores_df = scores_df[['date', 'smape'] + [col for col in scores_df.columns if col not in ['date', 'smape']]]
    outfile_name = 'shortcut_mlp_experiment_{}_{}.csv'.format(num, city)
    outfile_path = '../summaries/{}'.format(outfile_name)
    if os.path.exists(outfile_path):
        df = pd.read_csv(outfile_path)
        scores_df = pd.concat([scores_df, df], axis=0)
        scores_df.drop_duplicates(inplace=True)
        scores_df.sort_values(by='date', inplace=True)
    scores_df.to_csv('../summaries/{}'.format(outfile_name), index=False)
    with open('../summaries/shortcut_mlp_settings.txt', 'a') as f:
        f.write(str(setting) + ' Summary File: ' + outfile_name)
    vprint(1, '# ---- mean {} '.format(scores_df['smape'].mean()))


def fit_predict(city, sub, w_train_data, train_data, train_split_date, history_length, windows, pred_date, dropout_rate,
                units, batch_size, l2_strength=0.0001, n_folds=5, vprint=print):
    verbose_level = getattr(vprint, 'verbose_level', 500)

    x_train, y_train = wide_make_fw_x_y(
        wdata=w_train_data, ldata=train_data, split_date=train_split_date, history_length=history_length, num_shifts=60,
        use_medians=True, median_shifts=2, window_sizes=windows, for_prediction=False, n_thread=8, vprint=vprint,
        save_feature=True, use_cache=True, window_name='bj_pred_train'
    )

    x_test, _ = wide_make_fw_x_y(
        wdata=w_train_data, ldata=train_data, split_date=pred_date, history_length=history_length, num_shifts=1,
        use_medians=True, median_shifts=2, window_sizes=windows, for_prediction=True, n_thread=8, vprint=vprint,
        save_feature=True, use_cache=True, window_name='bj_pred_test'
    )

    x_train = x_train.loc[x_train['stationId'].map(lambda x: x.split('#')[0] != 'zhiwuyuan_aq')]
    y_train = y_train.loc[y_train['stationId'].map(lambda x: x.split('#')[0] != 'zhiwuyuan_aq')]

    subs = []
    min_valid_smape = []

    groups = x_train['stationId'].map(lambda x: x.split('#')[0])
    group_kfold = GroupKFold(n_splits=n_folds)
    splits = list(group_kfold.split(X=x_train, groups=groups))

    for it, (train_idx, val_idx) in enumerate(splits):
        vprint(2, '# ---- fold {} ----'.format(it + 1))
        model = get_keras_model(input_dim=x_train.shape[1] - 1, dropout_rate=dropout_rate, l2_strength=l2_strength, units=units)
        if it == 0:
            vprint(1, model.summary())

        history = model.fit(
            x=x_train.iloc[train_idx, 1:].values,
            y=y_train.iloc[train_idx, 1:].values,
            validation_data=(x_train.iloc[val_idx, 1:].values, y_train.iloc[val_idx, 1:].values),
            batch_size=batch_size,
            epochs=65535,
            verbose=verbose_level,
            callbacks=[
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=verbose_level),
                EarlyStopping(monitor='val_loss', patience=30, verbose=verbose_level),
                ModelCheckpoint(filepath='./model_checkpoint_{}_{}.hdf5'.format(city, pred_date), monitor='val_loss',
                                save_best_only=True, save_weights_only=True, mode='min')
            ])

        min_valid_smape.append(np.min(history.history['val_loss']))

        predictions = model.predict(x_test.iloc[:, 1:], verbose=verbose_level)
        predictions = pd.DataFrame(predictions)
        predictions['stationId'] = x_test['stationId'].map(lambda x: x.split('#')[0]).tolist()
        predictions['measure'] = x_test['stationId'].map(lambda x: x.split('#')[1]).tolist()

        vprint(2, '# ---- formatting submission df ----')
        for idx, row in tqdm(predictions.iterrows()):
            values = row[:48].values
            sub.loc[sub.test_id.isin([row['stationId'] + '#' + str(i) for i in range(48)]), row['measure']] = values
        subs.append(sub[SUB_COLS].copy())

    vprint(2, 'mean {}, std {}'.format(np.mean(min_valid_smape), np.std(min_valid_smape)))

    submissions = subs[0]
    for sub in subs[1:]:
        submissions[['PM2.5', 'PM10', 'O3']] += sub[['PM2.5', 'PM10', 'O3']]
    submissions[['PM2.5', 'PM10', 'O3']] /= n_folds
    return submissions


def predict(*, pred_date: str, bj_his_length=360, ld_his_length=420, bj_windows='golden_8', ld_windows='fib_8',
            bj_dropout=0.6, ld_dropout=0.2, bj_units=(48, 48, 48, 48),  ld_units=(24, 24, 24, 24), bj_batchsize=84,
            ld_batchsize=22, verbose: int=2, save=True, dosubmit=False, suffix='alt_lgb_split'):
    vprint = get_verbose_print(verbose_level=verbose)
    pred_date = pd.to_datetime(pred_date)
    get_new_data = pred_date > pd.to_datetime('2018-03-28')
    sub = pd.read_csv("../input/sample_submission.csv")
    OUTDIR = '../submission/sub_{}-{}-{}'.format(pred_date.year, pred_date.month, pred_date.day)
    os.system('mkdir -p {}'.format(OUTDIR))
    predict_start_day = pred_date + pd.Timedelta(1, unit='D')
    predict_start = pd.to_datetime(get_date(predict_start_day))

    bj_data = get_city_data(city='bj', vprint=vprint, impute_with_lgbm=False, partial_data=False, get_new_data=get_new_data)
    ld_data = get_city_data(city='ld', vprint=vprint, impute_with_lgbm=False, partial_data=False, get_new_data=get_new_data)

    vprint(2, bj_data.head())
    vprint(2, bj_data.loc[bj_data['stationId']!= 'zhiwuyuan_aq'].tail())
    vprint(2, ld_data.head())
    vprint(2, ld_data.tail())

    bj_data = impute(bj_data, lgbm=True, hour=True, mean=True)
    ld_data = impute(ld_data, lgbm=True, hour=True, mean=True)

    vprint(2, bj_data.head())
    vprint(2, bj_data.loc[bj_data['stationId']!= 'zhiwuyuan_aq'].tail())
    vprint(2, ld_data.head())
    vprint(2, ld_data.tail())

    bj_w_train_data = long_to_wide(bj_data)
    ld_w_train_data = long_to_wide(ld_data)

    train_split_date = pred_date - pd.Timedelta(3, unit='D')
    bj_pred = fit_predict(city='bj', sub=sub, w_train_data=bj_w_train_data, train_data=bj_data,
                          train_split_date=train_split_date, history_length=bj_his_length, pred_date=pred_date,
                          windows=MEDIAN_WINDOWS[bj_windows], dropout_rate=bj_dropout, units=bj_units,
                          batch_size=bj_batchsize, l2_strength=0.0001, n_folds=5, vprint=vprint
    )

    ld_pred = fit_predict(city='ld', sub=sub, w_train_data=ld_w_train_data, train_data=ld_data,
                          train_split_date=train_split_date, history_length=ld_his_length, pred_date=pred_date,
                          windows=MEDIAN_WINDOWS[ld_windows], dropout_rate=ld_dropout, units=ld_units,
                          batch_size=ld_batchsize, l2_strength=0.0001, n_folds=5, vprint=vprint
    )

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
            filepath = '{}/model_{}_sub.csv'.format(OUTDIR, 6)
        else:
            filepath = '{}/model_{}_sub_{}.csv'.format(OUTDIR, 6, suffix)
        submissions.to_csv(filepath, index=False)

        if dosubmit:
            submit(subfile=filepath,
                   description='model_{}_{}'.format(6, str(predict_start).split()[0]),
                   filename='model_{}_sub_{}.csv'.format(6, str(predict_start).split()[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment or predicting with nn model')
    parser.add_argument('--mode', type=str, default='exp', help='whether to do experiment or make prediction')
    parser.add_argument('--num', type=int, default=0, help='experiment to run')
    parser.add_argument('--exp_start_date', type=str, default='2018-03-21', help='date to start experiment, inclusive')
    parser.add_argument('--exp_end_date', type=str, default='2018-05-08', help='date to end experiment, exclusive')
    parser.add_argument('--pred_date', type=str, default=None, help='to generate prediction for this date')
    parser.add_argument('--pred_bj_his_length', type=int, default=360, help='number of days of history to use for bj')
    parser.add_argument('--pred_ld_his_length', type=int, default=420, help='number of days of history to use for ld')
    parser.add_argument('--pred_bj_dropout', type=float, default=0.6, help='regularization param bj')
    parser.add_argument('--pred_ld_dropout', type=float, default=0.2, help='regularization param ld')
    parser.add_argument('--pred_bj_batchsize', type=int, default=84, help='batch size param bj')
    parser.add_argument('--pred_ld_batchsize', type=int, default=22, help='batch size param ld')
    parser.add_argument('--pred_bj_windows', type=str, default='golden_8', help='window sizes for bj median feature')
    parser.add_argument('--pred_ld_windows', type=str, default='fib_8', help='window sizes for ld median feature ')
    parser.add_argument('--save', type=str2bool, default='True', help='whether to save submission file')
    parser.add_argument('--save_suffix', type=str, default='alt_lgb_fin', help='suffix append to submission filename')
    parser.add_argument('--submit', type=str2bool, default='False', help='whether to submit submission file')
    parser.add_argument('--cpu_only', type=str2bool, default='False', help='whether to only use cpu')
    parser.add_argument('--verbose', type=int, default=2, help='verbose level')
    args = parser.parse_args()
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.mode == 'exp':
        start_date = pd.to_datetime(args.exp_start_date)
        end_date = pd.to_datetime(args.exp_end_date)
        settings = [
            Setting(num=0, city='ld', history_length=420, num_shifts=60, median_shifts=2, median_windows='fib_8',
                    dropout_rate=0.2, l2_strength=0.0001, units=[24, 24, 24, 24], batch_size=22),
            Setting(num=1, city='bj', history_length=360, num_shifts=60, median_shifts=2, median_windows='golden_8',
                    dropout_rate=0.6, l2_strength=0.0001, units=[48, 48, 48, 48], batch_size=84)
        ]
        for setting in settings:
            if setting.num == args.num:
                run_setting(setting=setting, start_date=start_date, end_date=end_date, skip=1)
    else:
        if not args.pred_date
            args.pred_date = get_date(pd.to_datetime(datetime.now()))
        predict(pred_date=args.pred_date, bj_his_length=args.pred_bj_his_length, ld_his_length=args.pred_ld_his_length,
                bj_dropout=args.pred_bj_dropout, ld_dropout=args.pred_ld_dropout, bj_batchsize=args.pred_bj_batchsize,
                ld_batchsize=args.pred_ld_batchsize, bj_windows=args.pred_bj_windows, ld_windows=args.pred_ld_windows,
                save=args.save, dosubmit=args.submit, suffix=args.save_suffix, verbose=args.verbose)
