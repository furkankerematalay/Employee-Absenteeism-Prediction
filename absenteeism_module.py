import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# --- ASIL MOTOR SINIFI ---
class absenteeism_model():

    def __init__(self, model_file, scaler_file):
        # Dosyaları yükle
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None

    # Veriyi temizleme fonksiyonu (Notebook'ta yaptıklarının aynısı)
    def load_and_clean_data(self, data_file):

        # CSV'yi oku
        df = pd.read_csv(data_file)

        # Eğer dosyada 'Unnamed: 0' gibi gizli bir indeks sütunu varsa onu hemen atıyoruz.
        columns_to_drop = ['ID', 'Unnamed: 0', 'Absenteeism Time in Hours']
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
        # Orijinalini sakla
        self.df_with_predictions = df.copy()

        df['Reason for Absence'] = df['Reason for Absence'].fillna(0)  # Nan varsa 0 yap

        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)

        # Sütunları grupla
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

        df = df.drop(['Reason for Absence'], axis=1)

        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                        'Pets', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names

        reordered_columns = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                             'Date', 'Transportation Expense', 'Distance to Work', 'Age',
                             'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets']
        df = df[reordered_columns]

        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)
        df['Month Value'] = list_months

        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())

        df = df.drop(['Date'], axis=1)

        df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})
        df['Education'] = df['Education'].fillna(0)  # Garanti olsun

        # Gereksizleri at (Backward Elimination)
        df = df.drop(['Distance to Work', 'Day of the Week', 'Daily Work Load Average'], axis=1)

        # --- Nans kontrolü ---
        df = df.fillna(value=0)



        columns_to_drop_step2 = ['Day of the Week', 'Daily Work Load Average', 'Distance to Work']

        # Hata vermesin diye 'errors=ignore' ekliyoruz, belki daha önce silmişsindir.
        df = df.drop(columns_to_drop_step2, axis=1, errors='ignore')

        reordered_columns = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                             'Month Value', 'Transportation Expense', 'Age',
                             'Body Mass Index', 'Education', 'Children', 'Pets']

        df = df[reordered_columns]

        self.preprocessed_data = df.copy()

        # Ölçekle
        self.data = self.scaler.transform(df)

    # Sonucu tahmin etme fonksiyonu
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data