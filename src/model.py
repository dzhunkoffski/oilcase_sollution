import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
import ruptures as rpt
import src.physical_features as physical_features
import src.cpd as cpd

class SolutionModel:
    def __init__(self, segment_classifier: callable = None):
        self.classifier = segment_classifier
        if self.classifier is None:
            self.classifier = CatBoostClassifier(
                iterations=1750,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                learning_rate=0.01,
                random_seed=42
            )

    def pickle2dataframe(self, data_path: str):
        with open(data_path, 'rb') as fd:
            data = pickle.load(fd)
        df_list = []
        for key in data.keys():
            df = data[key]
            df['well_id'] = key
            df_list.append(df)
        return pd.concat(df_list)

    def fit(self, train_data_path: str):
        # Get dataframe from pickle file
        np.random.seed(42)
        train_timeseries = self.pickle2dataframe(data_path=train_data_path)

        # Get features
        train_data = physical_features.get_dataframe(data=train_timeseries, train=True)
        train_data = train_data.reset_index(drop=True)
        train_data['facie_code'] = train_data['facie_code'].astype(int)
        drop_inds = train_data[train_data['facie_code'] == -1].index
        drop_inds = np.random.choice(drop_inds, size=len(drop_inds) // 10 * 4)
        train_data = train_data.drop(drop_inds)
        X_train, y_train = train_data[['G', 'S', 'P', 'Am', 'am', 'Kp', 'R']], train_data['facie_code']
        self.classifier.fit(X_train, y_train)
        return self
    
    def predict(self, test_data_path: str):
        test_timeseries = self.pickle2dataframe(data_path=test_data_path)

        # Get features
        test_data = physical_features.get_dataframe(data=test_timeseries, train=False)
        X_test = test_data[['G', 'S', 'P', 'Am', 'am', 'Kp', 'R', 'meanSP', 'minSP', 'maxSP']]
        y_pred = self.classifier.predict(X_test).flatten()
        test_data['predicted_facie'] = y_pred
        test_timeseries = cpd.numerate_facie(df=test_timeseries, train=False)
        test_data['facie_id'] = test_data['facie_id'].astype(int)
        merged_df = test_timeseries.join(test_data[['facie_id', 'predicted_facie']], on=['facie_id'], how='left', rsuffix='_r')
        merged_df['predicted_facie'] = merged_df['predicted_facie'].astype(int)
        return merged_df[['well_id', 'predicted_facie']]
    
    def eval(self, test_data_path: str):
        # dataframes should contains "Facie_code"
        test_timeseries = self.pickle2dataframe(data_path=test_data_path)
        y_true = cpd.numerate_facie(df=test_timeseries, train=True)['Facie_code'].astype(int)
        y_pred = self.predict(test_data_path=test_data_path)['predicted_facie']

        print(f'F1@8: {np.sort(f1_score(y_true, y_pred, average=None))[::-1][:8].mean()}')
        print(f'F1: {np.sort(f1_score(y_true, y_pred, average=None)).mean()}')

    def build_sollution(self, test_data_path: str, save_to_path: str):
        prediction = self.predict(test_data_path=test_data_path)
        with open(test_data_path, 'rb') as fd:
            sollution_data = pickle.load(fd)
        for well_id in sollution_data.keys():
            sollution_data[well_id]['PREDICT'] = prediction[prediction['well_id'] == well_id]['predicted_facie']
            sollution_data[well_id] = sollution_data[well_id].drop(columns=['MD'])
            sollution_data[well_id]['PREDICT'] = sollution_data[well_id]['PREDICT'].astype(int)
            sollution_data[well_id]['GR_log'] = sollution_data[well_id]['GR_log'].astype(float)
            sollution_data[well_id]['SP_log'] = sollution_data[well_id]['SP_log'].astype(float)
        with open(save_to_path, 'wb') as fd:
            pickle.dump(sollution_data, fd)