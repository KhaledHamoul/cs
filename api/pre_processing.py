import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
from matplotlib.pyplot import figure

class pre_processing:

    def __init__(self, data):
        self.data = data

    def missing_percent(self):
        missing_col = list(self.data.isna().sum() != 0)

        try:
            if True not in missing_col:
                return "There is no missing values."

            self.data = self.data.loc[:, missing_col]
            missing_percent = (self.data.isna().sum() /
                               self.data.shape[0]) * 100

            df = pd.DataFrame()
            df['Total'] = self.data.isna().sum()
            df['perc_missing'] = missing_percent

            fig = figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
            left = 0.05
            bottom = 0.25
            width = 0.9
            height = 0.65
            fig.add_axes([left, bottom, width, height])

            p = sns.barplot(x=df.perc_missing.index, y='perc_missing', data=df)
            plt.ylabel('- Missing values (%) -')
            plt.xlabel('- Dataset attributes -')
            plt.xticks(rotation=80)
            p.tick_params(labelsize=12)

            # avoid circular imports (error)
            from api.helpers import getPltImage

            visual = '<div class="col-sm-12">' + getPltImage(plt) + '</div>'
            visual = '<div class="row">' + visual + '</div>'
        except:
            return'There is no missing values.'
        return visual, df.sort_values(ascending=False, by='Total', axis=0).to_json()

    # link : https://www.kaggle.com/smohubal/market-customer-segmentation-clustering/notebook
    def reduce_mem_usage(self, verbose=True):

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = self.data.memory_usage().sum() / 1024**2  # Memory total(Ram)

        for col in tqdm(self.data.columns):
            col_type = self.data[col].dtypes

            if col_type in numerics:
                c_min = self.data[col].min()
                c_max = self.data[col].max()

                # Int
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.data[col] = self.data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.data[col] = self.data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.data[col] = self.data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.data[col] = self.data[col].astype(np.int64)

                # Float
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.data[col] = self.data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.data[col] = self.data[col].astype(np.float32)
                    else:
                        self.data[col] = self.data[col].astype(np.float64)

        end_mem = self.data.memory_usage().sum() / 1024**2
        if verbose:
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
                end_mem, 100 * (start_mem - end_mem) / start_mem))
        return self.data

    def value_symmetry(self, target):
        return self.data[target].value_counts().plot('bar')

    def kde_plots(self, columns: list, hue_col: str):

        for c in columns:
            # hue loop
            for hue_value in self.data[hue_col].unique():
                sns.distplot(
                    self.data[self.data[hue_col] == hue_value][c], hist=False, label=hue_value)
            plt.show()

    def plots(self, columns: list, hue_col):
        _, axs = plt.subplots(
            int(round(len(columns) / 2, 0)), 5, figsize=(12, 12))

        for n, c in enumerate(columns):
            # hue loop
            for hue_value in self.data[hue_col].unique():
                sns.distplot(self.data[self.data[hue_col] == hue_value]
                             [c], hist=False, label=hue_value, ax=axs[n//5][n % 5])
            plt.tight_layout()
        plt.show()
