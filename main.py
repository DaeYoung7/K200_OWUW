from data_manager import *
from learning import *

import argparse

bm_name = 'K200'
from sklearn.preprocessing import MinMaxScaler
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, data_all_day = read_data(bm_name)
    ret, label = make_label(data, data_all_day, bm_name)
    data, bm = preprocessing(data, data_all_day, bm_name)
    data = get_stationary(data)
    # data에는 최근 데이터까지 포함, 최근 한달 데이터는 label값이 없음(1달 수익률 구할 수 없기 때문)
    data = data.loc[label.index]

    print('[Complete data managing]')
    print(f'data shape : {data.shape}')
    print('\n[Start Learning]')
    y_pred, test_label, dates = realtime_test(data, label, bm)
    pred_label = save_result(y_pred, test_label, dates, bm_name)
    analyzing(y_pred, test_label)
    print('[Finish Learning]')
