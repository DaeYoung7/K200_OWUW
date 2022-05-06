import pandas as pd

from data_manager import *
from learning import *

import argparse

bm_name = 'K200'
is_quantile = False

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # data, data_all_day = read_data(bm_name)
    # ret, label = make_label(data, data_all_day, bm_name)
    # data, bm = preprocessing(data, data_all_day, bm_name)
    # data = get_stationary(data)
    # # data에는 최근 데이터까지 포함, 최근 한달 데이터는 label값이 없음(1달 수익률 구할 수 없기 때문)
    # data = data.loc[label.index]
    data = pd.read_csv('data/data_preprocessed.csv', index_col='date', parse_dates=True)
    label = pd.read_csv('data/label.csv', index_col='date', parse_dates=True)
    ret = pd.read_csv('data/return.csv', index_col='date', parse_dates=True)
    print('[Complete data managing]')
    print(f'data shape : {data.shape}')
    print('\n[Start Learning]')
    result_data = realtime_test(data, label, ret, is_quantile)
    analyzing(result_data['pred'], result_data['label'], 'lstm')
    print('[Finish Learning]')
