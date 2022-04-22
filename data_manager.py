from utils import *


# read data and make X,y
def read_data(bm):
    data = pd.read_excel('data/'+bm+'.xlsx', sheet_name='Data', index_col='date')
    macro_data_all_day = pd.read_excel('data/'+bm+'.xlsx', sheet_name='Macro', header=13)
    macro_data_all_day.columns = ['date'] + list(data.columns[3:])
    macro_data_all_day = macro_data_all_day.set_index('date')
    price_data_all_day = pd.read_excel('data/'+bm+'.xlsx', sheet_name='Price', header=13)
    price_data_all_day.columns = ['date'] + list(data.columns[:3])
    price_data_all_day = price_data_all_day.set_index('date')
    data_all_day = pd.concat([price_data_all_day, macro_data_all_day], axis=1)

    # 거래대금 제외
    data = data.drop(['Kospi_TV'], axis=1)
    data_all_day = data_all_day.drop(['Kospi_TV'], axis=1)
    return data, data_all_day.ffill()


def make_label(data, price_data, bm):
    delete_last_1M = data[data.index < data.index[-1] - pd.DateOffset(months=2)].index
    future_idx = delete_last_1M + pd.DateOffset(months=2)
    future_price = price_data[bm].loc[future_idx].copy()
    future_price.index = delete_last_1M
    ret_1m = future_price / data[bm].loc[delete_last_1M] - 1
    label = ret_1m.copy()
    label[label > 0.0] = 1.0
    label[label < 0.0] = 0.0
    return ret_1m, label


def preprocessing(data, data_all_day, bm_name):
    bm = data[bm_name]
    diff_cols = ['Foreign_Ratio', 'GDP_Deflator', 'Facility_I', 'Construction_I', 'Saving_Rate', 'KOR_Growth', 'US_Growwth']
    log_diff_cols = ['K200', 'GDP', 'GNP', 'CLI', 'CCI', 'IAIP', 'Construction_Order', 'BDI', 'Total_Index', 'PPI', \
                     'EPI', 'IPI','M1', 'M2', 'Saving_Deposit', 'Market_Interest', 'Gold', 'Dollar', 'Yen', 'US_Interest',\
                     'US_3M_Interest', 'US_1Y_Interest', 'S&P500', 'Nasdaq100', 'SOX', 'Taiwan']
    diff_data_M = diff(data[diff_cols], data_all_day[diff_cols], 1, False)
    diff_data_Q = diff(data[diff_cols], data_all_day[diff_cols], 3, False)
    diff_data_H = diff(data[diff_cols], data_all_day[diff_cols], 6, False)
    diff_data_Y = diff(data[diff_cols], data_all_day[diff_cols], 12, False)

    log_diff_data_M = diff(data[log_diff_cols], data_all_day[log_diff_cols], 1, True)
    log_diff_data_Q = diff(data[log_diff_cols], data_all_day[log_diff_cols], 3, True)
    log_diff_data_H = diff(data[log_diff_cols], data_all_day[log_diff_cols], 6, True)
    log_diff_data_Y = diff(data[log_diff_cols], data_all_day[log_diff_cols], 12, True)

    diff_data = pd.concat([diff_data_M, diff_data_Q, diff_data_H, diff_data_Y], axis=1)
    log_diff_data = pd.concat([log_diff_data_M, log_diff_data_Q, log_diff_data_H, log_diff_data_Y], axis=1)
    data = pd.concat([diff_data, log_diff_data], axis=1)

    return data, bm


def get_stationary(data):
    non_stationary = []
    for c in data.columns:
        if not is_stationary(data[c]):
            non_stationary.append(c)
    print(f'불안정시계열 : {non_stationary}')
    return data.drop(non_stationary,axis=1)