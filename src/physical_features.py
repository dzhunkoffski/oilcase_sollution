import numpy as np
import pandas as pd
import src.cpd as cpd

def G(a: np.array):
    n = len(a)
    return np.dot(a, np.arange(1, n + 1)) / (np.sum(a) * n)

def S(a: np.array):
    n = len(a)
    return np.linalg.norm(a - a.mean()) / np.sqrt(n)

# P will define in common function

def Am(a: np.array):
    return 0.5 * (np.mean(a) + np.median(a))

def am(a: np.array):
    return a.max() - a.min()

def Kp(a: np.array, d: np.array):
    if (((a - a.mean()) ** 2).sum()) == 0:
        return 0
    return ((a - a.mean()) * (d - d.mean())).sum() / (((a - a.mean()) ** 2).sum())

def R(a: np.array, A: np.array):
    n = len(a)
    return len(A) / n

def get_dataframe(data: pd.DataFrame, train: bool):
    df = cpd.numerate_facie(df=data, train=train)
    features = []
    for facie_id in df['facie_id'].unique():
        segment = df[df['facie_id'] == facie_id].reset_index(drop=True)
        if train:
            facie_code = segment['Facie_code'][0]
        well_id = segment['well_id'][0]
        facie_size = segment.shape[0]

        a = segment['GR_log']
        A = segment['GR_log'][(segment.GR_log.shift(1) < segment.GR_log) & (segment.GR_log.shift(-1) < segment.GR_log)]
        d = segment['MD'].values[a.index]
        D = segment['MD'].values[A.index]
        a = a.values
        A = A.values

        # EXtract features
        G_ = G(a)
        S_ = S(a)
        if segment['MD'].max() - segment['MD'].min() == 0:
            P_ = 0
        else:
            P_ = (np.abs(a.max() - np.mean(segment['SP_log']))) / (segment['MD'].max() - segment['MD'].min())
        Am_ = Am(a)
        am_ = am(a)
        Kp_ = Kp(a, d)
        R_ = R(a, A)
        meanSP_ = np.mean(segment['SP_log'])
        maxSP_ = np.max(segment['SP_log'])
        minSP_ = np.min(segment['SP_log'])
        if train:
            features.append([G_, S_, P_, Am_, am_, Kp_, R_, facie_id, well_id, facie_size, facie_code, meanSP_, maxSP_, minSP_])
        else:
            features.append([G_, S_, P_, Am_, am_, Kp_, R_, facie_id, well_id, facie_size, meanSP_, maxSP_, minSP_])
    if train:
        return pd.DataFrame(data=np.array(features), columns=['G', 'S', 'P', 'Am', 'am', 'Kp', 'R', 'facie_id', 'well_id', 'facie_size', 'facie_code', 'meanSP', 'maxSP', 'minSP'])
    return pd.DataFrame(data=np.array(features), columns=['G', 'S', 'P', 'Am', 'am', 'Kp', 'R', 'facie_id', 'well_id', 'facie_size', 'meanSP', 'minSP', 'maxSP'])

