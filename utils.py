import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def diff(data, data_all_day, term, log=True):
    diff_date = data.index - pd.DateOffset(months=term)
    data = data.copy()
    data_all_day = data_all_day.copy()
    data[data < 0.] = 0.
    data_all_day[data_all_day < 0.] =  0.

    diff_data = data_all_day.loc[diff_date]
    diff_data.index = data.index

    if log:
        ret_data = data / diff_data - 1
        col_max_val = ret_data.max()
        for c in col_max_val[col_max_val==np.inf].index:
            for i in ret_data[c][ret_data[c]==np.inf].index:
                ret_data[c].loc[i] = ret_data[c][:i][:-1].max()

    else:
        ret_data = data - diff_data
    ret_data_cols = []
    for c in data.columns:
        ret_data_cols.append(c + '_' + str(term))
    ret_data.columns = ret_data_cols

    return ret_data

def is_stationary(tseries):
    tseries = tseries.dropna()
    dftest = adfuller(tseries, autolag='AIC')
    p_value = dftest[1]
    if p_value > 0.05:
        ret = False
    else:
        ret = True
    return ret

def save_result(y_pred, test_label, bm_name, is_quantile):
    criterion = 0. if is_quantile else 0.5
    pred_label = np.array(y_pred.copy())
    y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
    pred_label[pred_label > criterion] = 1.0
    pred_label[pred_label < criterion] = 0.0
    y_pred['pred_label'] = pred_label
    test_label = pd.DataFrame(test_label, columns=['test_label'])
    result = pd.concat([y_pred, test_label], axis=1)
    result.to_csv('result/'+bm_name+'.csv')
    return pred_label

def analyzing(pred_label, test_label):
    print(accuracy_score(test_label, pred_label))
    print(confusion_matrix(test_label, pred_label, labels=[0, 1]))
    print(classification_report(test_label, pred_label, labels=[0,1]))
    return