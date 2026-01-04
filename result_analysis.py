import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import pandas as pd
import os
from scipy.stats import wilcoxon
pandas2ri.activate()

sk = importr('ScottKnottESD')

def median_iqr_from_csv(csv_path: str) -> dict:
    # 1. 读取 CSV
    df = pd.read_csv(csv_path, index_col=0)

    # 2. 计算分位数
    q1 = df.quantile(0.25)
    median = df.quantile(0.5)
    q3 = df.quantile(0.75)

    # 3. IQR
    iqr = q3 - q1

    # 4. 组装结果字典
    stats_dict = {
        col: (float(median[col]), float(iqr[col]))
        for col in df.columns
    }

    return stats_dict
def sk_rank_from_csv(csv_path: str) -> dict:
    """
    使用 Scott-Knott ESD 对 CSV 中的算法进行排序

    参数
    ----------
    csv_path : str
        CSV 文件路径，列为算法，行为多次实验结果

    返回
    ----------
    rank_dict : dict
        {algorithm_name: rank}
    """


    df = pd.read_csv(csv_path, index_col=0)

    r_sk = sk.sk_esd(df)

    column_order = list(r_sk[3])
    ranks = r_sk[1].astype(int)

    max_rank = ranks.max()
    reversed_ranks = max_rank - ranks + 1

    rank_dict = {
        df.columns[i - 1]: int(rank)
        for i, rank in zip(column_order, reversed_ranks)
    }

    return rank_dict

def wilcoxon_vs_dhda(
    csv_path: str,
    baseline: str = "DHDA",
    alpha: float = 0.05
) -> dict:

    df = pd.read_csv(csv_path, index_col=0)

    if baseline not in df.columns:
        raise ValueError(f"Baseline column '{baseline}' not found in CSV")

    baseline_data = df[baseline]

    result_dict = {}

    for alg in df.columns:
        if alg == baseline:
            continue

        other_data = df[alg]

        # 删除 NaN，保持配对
        paired = pd.concat([baseline_data, other_data], axis=1).dropna()
        x = paired.iloc[:, 0]
        y = paired.iloc[:, 1]

        # Wilcoxon 要求：样本数 > 0，且不全相等
        if len(x) == 0 or (x == y).all():
            result_dict[alg] = False
            continue
        d = np.around(x - y, decimals=10)
        res = wilcoxon(d)


        result_dict[alg] = res.pvalue < alpha

    return result_dict

def build_summary_table(
    rank_dict: dict,
    stats_dict: dict,
    float_fmt: str = ".2f"
) -> pd.DataFrame:
    """
    根据 rank 字典和 (median, IQR) 字典生成汇总表

    参数
    ----------
    rank_dict : dict
        {algorithm: rank}
    stats_dict : dict
        {algorithm: (median, iqr)}
    float_fmt : str
        浮点数格式，例如 ".4f"

    返回
    ----------
    summary_df : pd.DataFrame
        index 为算法名，单列 summary，格式为 rank_median_iqr
    """

    records = {}

    for alg in rank_dict:
        if alg not in stats_dict:
            raise KeyError(f"Algorithm '{alg}' missing in stats_dict")

        rank = rank_dict[alg]
        median, iqr = stats_dict[alg]

        records[alg] = f"{rank}_{median:{float_fmt}}_{iqr:{float_fmt}}"

    summary_df = pd.DataFrame.from_dict(
        records, orient="index", columns=["summary"]
    )

    return summary_df

RQs_result = {
    'RQ1': r'\results\RQ1-RESULT\MAPE',
    'RQ2': r'\results\RQ2-RESULT',
    'RQ3': r'\results\RQ3-RESULT',
    'RQ4': r'\results\RQ4-RESULT\MAPE'
}

for rq, result_path in RQs_result.items():
    print(f'---------{rq}---------')
    if rq == 'RQ1' or rq == 'RQ2':
        print(f'summary info: rank_median_iqr')
        for sys_file in os.listdir(result_path):
            print(sys_file)
            if sys_file.endswith('.csv'):
                file_path = os.path.join(result_path, sys_file)
                rank_dict = sk_rank_from_csv(file_path)
                stats_dict = median_iqr_from_csv(file_path)
                df = build_summary_table(rank_dict, stats_dict)
                print(df)
    elif rq == 'RQ3':
        print(f'summary info: stat-significance_median_iqr')
        for sys_file in os.listdir(result_path):
            print(sys_file)
            if sys_file.endswith('.csv'):
                file_path = os.path.join(result_path, sys_file)
                w_dict = wilcoxon_vs_dhda(file_path)
                stats_dict = median_iqr_from_csv(file_path)
                df = build_summary_table(w_dict, stats_dict)
                print(df)
    elif rq == 'RQ4':
        print(f'summary info: median-error_median-time')
        for sys_file in os.listdir(result_path):
            print(sys_file)
            mape_file_path = os.path.join(result_path, sys_file)
            time_file_path = mape_file_path.replace('MAPE', 'TIME')

            mape_stats_dict = median_iqr_from_csv(mape_file_path)
            time_stats_dict = median_iqr_from_csv(time_file_path)
            print(mape_stats_dict)
            print(time_stats_dict)






