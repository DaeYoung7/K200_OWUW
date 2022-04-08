from data_manager import *
from learning import *

bm_name = 'K200'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, data_all_day = read_data(bm_name)
    label = make_label(data, data_all_day, bm_name)
    data, bm = preprocessing(data, data_all_day, bm_name)
    data = get_stationary(data)
    print('[Complete data managing]')
    print(f'data shape : {data.shape}')
    print('\n[Start Learning]')
    y_pred, test_label = train_test(data, label, bm)
    print('[Finish Learning]')
    pred_label = save_result(y_pred, test_label, bm_name)
    analyzing(pred_label, test_label)