# coding=utf-8
import numpy as np
from math import log1p, pi, e, atan, exp, log
import scipy.special as special

k_max = 0.8038


def flat_head(x, floor, ceiling):
    x = min(x, ceiling)
    x = max(x, floor)
    return x


def max_min_normalization(x, min_value=0, max_value=1000):
    x = float(x - min_value) / float(max_value - min_value)
    # x.map({NaN:0})
    # x = pd.fillna(x([np.nan, 0], dtype=object))
    # where_are_nan = np.isnan(x)
    # x[where_are_nan] = 0
    return x


def price_bucket(x):
    if x > 518.57:
        return 1.0
    if 207.43 < x <= 518.57:
        return 0.875
    elif 103.7 < x <= 207.43:
        return 0.75
    elif 51.86 < x <= 103.7:
        return 0.625
    elif 20.74 < x <= 51.86:
        return 0.5
    elif 10.37 < x <= 20.74:
        return 0.375
    elif 5.19 < x <= 10.37:
        return 0.25
    else:
        return 0.125


def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - 0) / np.std(x)
    # x.map({'nan':1})
    # pd.isnull(x([np.nan, 0], dtype=object))
    where_are_nan = np.isnan(x)
    x[where_are_nan] = 0
    return x


def LogNormalization(x):
    """log normalization"""
    x = log1p(x)
    return x


def box_normalization(x, min_value=0, iqr=1):
    if x < iqr:
        x = float(x - min_value) / float(iqr - min_value)
        return k_max * x
    else:
        try:
            x = atan((pi * x) / iqr) * (2 / pi)
            return x
        except Exception as e:
            print('erroy is {} , date is {}'.format(e, x))
            return 0


def good_shelf_cycle(t):
    x = t
    # 按照下列标准进行分段
    if 0 <= x <= 3:
        return 1
    elif 3 < x <= 7:
        return 0.8
    elif 7 < x <= 14:
        return 0.7
    else:
        return 0.0


def CTR_segmentation(x, threshhold_1, threshhold_2):
    if x >= threshhold_1:
        return 1
    elif x <= threshhold_2:
        return 0
    else:
        return -1


def segmentation(x, threshhold):
    if x >= threshhold:
        return 1
    else:
        return 0


def get_threshold(df):
    """
    获取阀值
    :param df:DataFrame
    :return: 阀值
    """
    # 排序

    # 获取阀值

    return df


def inter_qr(df, k=1.5, q1=0.25, q2=0.75):
    """
    分箱模型
    :param df: Dataframe
    :param k: 异常度
    :param q1:第一分位点，默认为1/4点
    :param q3: 第三分位点，默认为3/4点
    :return: 异常阀值
    """
    qu1 = df.quantile(q1)
    qu2 = df.quantile(q2)
    iqr = qu2 - qu1
    outlier = qu2 + k * iqr
    print "第一分位点:", qu1
    print "第二分位点:", qu2
    print "异常阈值:", outlier
    return outlier


def gmatan_normalization(x, threshold=0):
    """
    毛利率归一化
    :param x: 每一个行元素
    :return: 处理后的元素
    """
    if x < threshold:
        return 0
    else:
        return x


def espboxNormalization(x, min_value, iqr):
    """
    曝光利润额归一化
    :param x: 每一个行元素
    :return: 处理后的元素
    """
    if x < 0.0:
        return 0.0
    elif x < iqr:
        x = float(x - min_value) / float(iqr - min_value)
        return k_max * x
    else:
        try:
            x = atan((pi * x) / iqr) * (2 / pi)
            return x
        except Exception as e:
            print('erroy is {} , date is {}'.format(e, x))
            return 0


# 评分归一化
def score_normalization(x):
    if x < 4:
        return x / 8.0
    else:
        return (x - 4) / 2.0 + 0.5


def wilson_score(pos, total, p_z=2.):
    """
    威尔逊得分计算函数
    :param pos: 正例数
    :param total: 总数
    :param p_z: 正太分布的分位数
    :return: 威尔逊得分
    """
    if total < 3000:
        total = 3000
    pos_rat = pos * 1. / total * 1.  # 正例比率
    score = (pos_rat + (np.square(p_z) / (2. * total))
             - ((p_z / (2. * total)) * np.sqrt(4. * total * (1. - pos_rat) * pos_rat + np.square(p_z)))) / \
            (1. + np.square(p_z) / total)
    return score


def Ctr_with_log_sigmoid(click, brower, avg_ctr):
    """
    利用激活函数sigmoid
    :param click:点击
    :param brower: 曝光
    :return: 得分
    """

    def sigmoid(gamma):
        if gamma < 0:
            return 1 - 1 / (1 + exp(gamma))
        else:
            return 1 / (1 + exp(-gamma))

    x = log((brower + 1), 2) - 10
    p = sigmoid(x)
    total = p * (click / float(brower)) + (1 - p) * avg_ctr
    return total


def use_bayes(imps, clks, iter_num, epsilon):
    alpha = 1
    beta = 1

    def fixed_point_iteration(imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))
        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)

    for i in range(iter_num):
        new_alpha, new_beta = fixed_point_iteration(imps, clks, alpha, beta)
        if abs(new_alpha - alpha) < epsilon and abs(new_beta - beta) < epsilon:
            break
        alpha = new_alpha
        beta = new_beta
    return alpha, beta