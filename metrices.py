import numpy as np

"""
使用下面的误差函数给定的参数为结果+预测得到的数据经过变化得出:
    // unnorm操作
    true -> targets.detach().cpu().numpy() * std + mean
    pred -> outputs.detach().cpu().numpy() * std + mean
"""


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    """
    mape: 1/n |pred - true| / true
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100  # use %


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    """
    rmse: \sqrt{1/n ((true - pred) ^ 2)}
    """
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.sqrt(np.mean(np.nan_to_num(mask * mse)))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    """
    mae: 1/n |true - pred|
    """
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))
