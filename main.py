from data_manager import *
from learning import *

import argparse

bm_name = 'K200'
from sklearn.preprocessing import MinMaxScaler
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version')
    parser.add_argument('--corr', action='store_true', default=False)
    args = parser.parse_args()
    # data, data_all_day = read_data(bm_name)
    # ret, label = make_label(data, data_all_day, bm_name)
    # data, bm = preprocessing(data, data_all_day, bm_name)
    # data = get_stationary(data)
    # # data에는 최근 데이터까지 포함, 최근 한달 데이터는 label값이 없음(1달 수익률 구할 수 없기 때문)
    # data = data.loc[label.index]
    data = pd.read_csv('data/data_preprocessed.csv', index_col='date', parse_dates=True)
    label = pd.read_csv('data/label.csv', index_col='date', parse_dates=True)
    bm = pd.read_csv('data/bm_data.csv', index_col='date', parse_dates=True)
    print('[Complete data managing]')
    print(f'data shape : {data.shape}')
    print('\n[Start Learning]')
    if args.version == '1':
        result_data = direct_ensemble(data, label, bm, args.corr)
        print(result_data)
        rf_pred = result_data['RF']
        rf_label = result_data['label']
    else:
        result_data = make_label_to_ensemble(data, label)
        result_data.to_csv('result/LR_SVC.csv')
        rf_pred, rf_label = ensemble(result_data)
    analyzing(result_data['LR'], result_data['label'], 'LR')
    analyzing(result_data['SVC'], result_data['label'], 'SVC')
    analyzing(rf_pred, rf_label, 'RF')
    print('[Finish Learning]')
