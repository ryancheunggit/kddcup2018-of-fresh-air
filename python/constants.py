#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
import pandas as pd

LONDON_STATIONS = ['BL0', 'CD1', 'CD9', 'GN0', 'GN3', 'GR4', 'GR9', 'HV1', 'KF1', 'LW2', 'MY7', 'ST5', 'TH4']
BEIJING_STATIONS = [
        'aotizhongxin_aq', 'badaling_aq', 'beibuxinqu_aq', 'dingling_aq', 'donggaocun_aq', 'dongsi_aq', 'dongsihuan_aq',
        'fangshan_aq', 'fengtaihuayuan_aq', 'guanyuan_aq', 'gucheng_aq', 'huairou_aq', 'liulihe_aq', 'mentougou_aq',
        'miyun_aq', 'miyunshuiku_aq', 'nansanhuan_aq', 'nongzhanguan_aq', 'pingchang_aq', 'pinggu_aq', 'qianmen_aq',
        'shunyi_aq', 'tiantan_aq', 'tongzhou_aq', 'wanliu_aq', 'wanshouxigong_aq', 'xizhimenbei_aq', 'yanqin_aq',
        'yizhuang_aq', 'yongdingmennei_aq', 'yongledian_aq', 'yufa_aq', 'yungang_aq', 'zhiwuyuan_aq', 'daxing_aq'
    ]

AQ_COL_NAMES = ['stationId', 'utc_time', 'PM2.5', 'PM10', 'O3']

BJ_STATION_CHN_PINGYING_MAPPING = {
        '东四': 'dongsi_aq', '天坛': 'tiantan_aq', '官园': 'guanyuan_aq', '万寿西宫': 'wanshouxigong_aq',
        '奥体中心': 'aotizhongxin_aq', '农展馆': 'nongzhanguan_aq', '万柳': 'wanliu_aq', '北部新区': 'beibuxinqu_aq',
        '植物园': 'zhiwuyuan_aq', '丰台花园': 'fengtaihuayuan_aq', '云岗': 'yungang_aq', '古城': 'gucheng_aq',
        '大兴': 'daxing_aq', '亦庄': 'yizhuang_aq', '通州': 'tongzhou_aq', '顺义': 'shunyi_aq', '昌平': 'pingchang_aq',
        '门头沟': 'mentougou_aq', '平谷': 'pinggu_aq', '怀柔': 'huairou_aq', '密云': 'miyun_aq', '延庆': 'yanqin_aq',
        '定陵': 'dingling_aq', '八达岭': 'badaling_aq', '密云水库': 'miyunshuiku_aq', '东高村': 'donggaocun_aq',
        '永乐店': 'yongledian_aq', '榆垡': 'yufa_aq', '琉璃河': 'liulihe_aq', '前门': 'qianmen_aq',
        '永定门内': 'yongdingmennei_aq', '西直门北': 'xizhimenbei_aq', '南三环': 'nansanhuan_aq',
        '东四环': 'dongsihuan_aq', '房山': 'fangshan_aq'
    }

MONTH_DIGIT_STR_MAPPING = {
        '01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'may', '06': 'jun', '07': 'jul', '08': 'aug',
        '09': 'sep', '10': 'oct', '11': 'nov', '12': 'dec'
    }

LONDON_API_COL_MAPPING = {
        'PM2.5 Particulate (ug/m3)': 'PM2.5',
        'PM10 Particulate (ug/m3)': 'PM10',
        'MeasurementDateGMT': 'utc_time'
    }

SUB_COLS = ['test_id', 'PM2.5', 'PM10', 'O3']

DATA_API_TOKEN = '2k0d1d8'

USERNAME = ''
SUB_TOKEN = ''

MEDIAN_WINDOW_SIZES = [11,  18,  30,  48,  78, 126, 203, 329]
MEDIAN_WINDOW_SIZES_FIB = [7, 14, 21, 35, 56, 91, 147, 238, 385]
GOLDEN_MEDIAN_WINDOWS = {'golden_{}'.format(i): MEDIAN_WINDOW_SIZES[:i] for i in range(4, 9)}
FIB_MEDIAN_WINDOWS = {'fib_{}'.format(i): MEDIAN_WINDOW_SIZES_FIB[:i] for i in range(5, 10)}
MEDIAN_WINDOWS = {**GOLDEN_MEDIAN_WINDOWS, **FIB_MEDIAN_WINDOWS}

BJ_HOLIDAYS = pd.concat([
    pd.DataFrame({
            'holiday': 'new_year',
            'ds': pd.to_datetime(['2017-01-02', '2018-01-01', '2018-01-02'])
        }),
    pd.DataFrame({
            'holiday': 'chinese_new_year',
            'ds': pd.to_datetime([
                '2017-01-27', '2017-01-28', '2017-01-29',
                '2018-02-15', '2018-02-16', '2018-02-17'
            ])
        }),
    pd.DataFrame({
            'holiday': 'labor_day',
            'ds': pd.to_datetime([
                '2017-05-01', '2017-05-02', '2017-05-03',
                '2018-05-01', '2018-05-02', '2018-05-03'
            ])
        }),
    pd.DataFrame({
            'holiday': 'qin_ming',
            'ds': pd.to_datetime([
                '2017-04-02', '2017-04-03', '2017-04-04',
                '2018-04-04', '2018-04-05', '2018-04-06'
            ])
        }),
    pd.DataFrame({
            'holiday': 'dragon_boat',
            'ds': pd.to_datetime(['2017-05-28', '2017-05-29', '2017-05-30'])
        }),
    pd.DataFrame({
            'holiday': 'golden_week',
            'ds': pd.to_datetime(['2017-10-01', '2017-10-02', '2017-10-03'])
        }),
    pd.DataFrame({
            'holiday': 'mid_autum',
            'ds': pd.to_datetime(['2017-10-04'])
        }),
    pd.DataFrame({
            'holiday': 'work_weekend',
            'ds': pd.to_datetime([
                '2017-01-22', '2017-02-04', '2017-04-01',
                '2017-05-27', '2017-07-30'
            ])
    })])
