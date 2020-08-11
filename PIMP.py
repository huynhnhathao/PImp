from scipy.stats.distributions import norm
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
from lightgbm import LGBMClassifier
import lightgbm as lgb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
%matplotlib inline

import warnings
warnings.simplefilter('ignore', UserWarning)

import gc
gc.enable()
from scipy.stats import norm

sns.set(style = 'whitegrid')
def my_gini(preds, y_true):
    print(type(preds))
    print(type(y_true))
    return ('Gini',2*roc_auc_score(y_true, preds) - 1, True)
class PIMP():
    def __init__(self):
        pass
    def get_feature_importance(self, data,y, shuffle, num_boost_round = 200, 
                               boosting_type = 'rf', sub_sample = 0.623, colsample_bytree = 0.7, num_leaves = 127, max_depth = 8, seed = None, bagging_freq = 1):
        # Gather real features
        train_features = [f for f in data if f != y ]

        # Shuffle target if required
        if shuffle:
            # Here you could as well use a binomial distribution
            label = data[y].copy().sample(frac=1.0)
        else:
            label = data[y]

        # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
        dtrain = lgb.Dataset(data[train_features], label, free_raw_data=False, silent=True)
        lgb_params = {
            'objective': 'binary',
            'boosting_type': boosting_type,
            'subsample': sub_sample,
            'colsample_bytree': colsample_bytree,
            'num_leaves': num_leaves,
            'max_depth':max_depth,
            'seed': None,
            'bagging_freq': bagging_freq,
            'n_jobs': 4
        }
    

        # Fit the model using cross validation
        clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round= num_boost_round)
                     

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        imp_df['Gini'] = 2*roc_auc_score(label, clf.predict(data[train_features])) - 1

        return imp_df
    
    

    
    def get_null_distribution(self,data, y, num_iter):
        null_df = pd.DataFrame()

        start = time.time()
        dsp = ''
        for i in range(num_iter):
            imp_df = self.get_feature_importance(data = data, y = y, shuffle = True)
            imp_df['run'] = i+1
            null_df = pd.concat([null_df, imp_df],axis = 0)
            # Erase previous message
            for l in range(len(dsp)):
                print('\b', end='', flush=True)
            # Display current run and time used
            spent = (time.time() - start) / 60
            dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, num_iter, spent)
            print(dsp, end='', flush=True)
        return null_df
    def get_actual_imp(self, data, y):
        imp_df = self.get_feature_importance(data, y, shuffle = False)
        return imp_df
    def get_actual_distribution(self, data, y, num_iter):
        actual_df = pd.DataFrame()

        start = time.time()
        dsp = ''
        for i in range(num_iter):
            imp_df = self.get_feature_importance(data = data, y = y, shuffle = False)
            imp_df['run'] = i+1
            actual_df = pd.concat([actual_df, imp_df],axis = 0)
            # Erase previous message
            for l in range(len(dsp)):
                print('\b', end='', flush=True)
            # Display current run and time used
            spent = (time.time() - start) / 60
            dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, num_iter, spent)
            print(dsp, end='', flush=True)
        return actual_df
    def scoring_features(self, null_imp_df, actual_imp_df):
        #this function take the log of (actual_imp / 75 percentile of null distribution) for each feature
        feature_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
            gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
            f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
            split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
            feature_scores.append((_f, split_score, gain_score))

        scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

        plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()
        return scores_df
    def correlation_score(self, null_imp_df, actual_imp_df):
        correlation_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature', data=corr_scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature', data=corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
        fig.subplots_adjust(top=0.93)
    def display_distributions(self, actual_imp_df_, null_imp_df_, feature_):
        plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, label='Null importances', bins = 20)
        ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(), 
                   ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
        ax.legend()
        ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
        plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances', bins = 20)
        ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(), 
                   ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
        ax.legend()
        ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
        plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())
    def t_test(self, null_imp, actual_imp, n):
        def pvalue(zscore):
            return 1-norm.cdf(zscore)
        mean_std = []
        #maximum likelihood estimation for normal distribution
        for col in actual_imp['feature']:
            mean_gain, std_gain = norm.fit(null_imp.loc[null_imp['feature'] == col, 'importance_gain'].values)
            mean_split, std_split = norm.fit(null_imp.loc[null_imp['feature'] == col, 'importance_split'].values)
            mean_std.append([mean_gain, std_gain, mean_split, std_split])
        df = pd.DataFrame()
        df['feature'] = actual_imp['feature']
        df = pd.concat([df, pd.DataFrame(mean_std, columns  = ['mean_gain', 'std_gain', 'mean_split', 'std_split'])], axis = 1)
        #compute t statistic and p-value of the actual imp wrt the null distribution
        df['gain_z_score'] = (actual_imp['importance_gain'] - df['mean_gain'])/ df['std_gain']

        df['gain_p_value'] = df['gain_z_score'].apply(pvalue)
        df['split_z_score'] = (actual_imp['importance_split'] - df['mean_split']) / df['std_split']
        df['split_p_value'] = df['split_z_score'].apply(pvalue)
        return df
