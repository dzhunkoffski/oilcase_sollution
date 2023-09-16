import numpy as np
import ruptures as rpt
import pandas as pd

def detect(ts: np.array):
    cpd = rpt.Pelt(model='l2', min_size=3, jump=3).fit(ts)
    cps = cpd.predict(pen=65)
    if cps[-1] == len(ts):
        cps = cps[:-1]
    cps = np.array(cps)
    cps = len(ts) - 1 - cps
    cps = np.sort(cps)
    cps = np.array([0] + list(cps) + [len(ts)])
    return cps

def extract_cp_ixs(cls: np.array):
    ixs = []
    for i in range(1, len(cls)):
        if cls[i-1] != cls[i]:
            ixs.append(i)
    ixs = [0] + ixs + [len(cls)]
    return ixs

def numerate_facie(df: pd.DataFrame, train: bool):
    df = df.copy()
    df_list = []
    id_cnt = 0
    for well_id in df['well_id'].unique():
        subdf = df[df['well_id'] == well_id]
        ts = subdf['GR_log'].values[::-1]
        if not train:
            change_points = detect(ts=ts)
        else:
            change_points = extract_cp_ixs(subdf['Facie_code'].values)
        subdf['facie_id'] = id_cnt
        if len(change_points) == 0:
            id_cnt += 1
        else:
            for i in range(1, len(change_points)):
                subdf.iloc[change_points[i-1]:change_points[i], subdf.columns.get_loc('facie_id')] = id_cnt
                id_cnt += 1
        df_list.append(subdf)
    return pd.concat(df_list)