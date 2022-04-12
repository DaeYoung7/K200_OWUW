from data_manager import *
from learning import *

bm_name = 'K200'
is_quantile = True
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
    y_pred, test_label = train_test(data, ret, label, bm, is_quantile)
    print('[Finish Learning]')
    pred_label = save_result(y_pred, test_label, bm_name)
    analyzing(pred_label, test_label)