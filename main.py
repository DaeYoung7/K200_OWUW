from data_manager import *
from learning_val import *

bm_name = 'K200'
is_quantile = True
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, data_all_day = read_data(bm_name)
    ret, label = make_label(data, data_all_day, bm_name)
    data, bm = preprocessing(data, data_all_day, bm_name)
    data = get_stationary(data)
    print('[Complete data managing]')
    print(f'data shape : {data.shape}')
    print('\n[Start Learning]')
    train_test(data, ret, label, bm, is_quantile)
    print('[Finish Learning]')
    # pred_label = save_result(y_pred, test_label, bm_name)
    # analyzing(pred_label, test_label)