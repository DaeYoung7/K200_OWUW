from data_manager import *
from learning_kfold import *

bm_name = 'K200'
is_quantile = True
num_fold = 3
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data, data_all_day = read_data(bm_name)
    ret, label = make_label(data, data_all_day, bm_name)
    data, bm = preprocessing(data, data_all_day, bm_name)
    data = get_stationary(data)

    # data에는 label을 알 수 없는 최근 데이터도 포함(1달 수익률이라 최근 1달 값은 알 수가 없음
    data = data.loc[label.index]
    print('[Complete data managing]')
    print(f'data shape : {data.shape}')
    print('\n[Start Learning]')
    kfold(data, ret, label, num_fold, is_quantile)
    print('[Finish Learning]')
    # pred_label = save_result(y_pred, test_label, bm_name)
    # analyzing(pred_label, test_label)