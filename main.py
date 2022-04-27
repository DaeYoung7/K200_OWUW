from data_manager import *
from learning import *

import argparse

bm_name = 'K200'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=False, action='store_true')  # true - test, false - kfold
    parser.add_argument('--quantile', default=False, action='store_true')  # true - quantile, false - binary classification
    parser.add_argument('--ts_layer', required=True)  # transformer or lstm
    args = parser.parse_args()

    data, data_all_day = read_data(bm_name)
    ret, label = make_label(data, data_all_day, bm_name)
    bm = data[bm_name]
    input_data = pd.read_csv('data/K200_deep_factor.csv', index_col='date', parse_dates=True)
    print('[Complete data managing]')
    print(f'data shape : {data.shape}')
    print('\n[Start Learning]')
    if args.test is True:
        y_pred, test_label = realtime_test(input_data, ret, label, bm, args.ts_layer, args.quantile)
        pred_label = save_result(y_pred, test_label, bm_name, args.quantile)
        analyzing(pred_label, test_label)
    else:
        num_fold = 5
        purged_kfold(input_data, ret, label, num_fold, args.ts_layer, args.quantile)
    print('[Finish Learning]')
