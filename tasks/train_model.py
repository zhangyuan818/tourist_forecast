#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
 @version: ??
 @author: ZhangYuan
 @file: train_model.py
 @time: 2019/10/22 14:48
"""
import matplotlib.pyplot as plt
import seaborn as sns

# modelling
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score,cross_val_predict,KFold
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,StandardScaler

def main():
    df = pd.read_csv("../data/featureData.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    df.head()
    # Explore feature distibution
    # 训练集与测试集各特征分布对比
    # plt.figure()
    # i=1
    # for column in df.columns[3:]:
    #     plt.subplot(3, 4, i)
    #     i=i+1
    #     g = sns.kdeplot(df[column]['2016':'2017'],color="blue",shade=True)
    #     g = sns.kdeplot(df[column]['2018'],ax = g, color="red",shade=True)
    #     g.set_xlabel(column)
    #     g.set_ylabel("Frequency")
    #     g = g.legend(["train","test"])
    #     plt.show()

    # #各特征在节假日不同时的分布对比
    # for i in range(3,len(df.columns)):
    #     if not i==10 and not i==11:
    #         g = sns.FacetGrid(df,col = 'holiday')
    #         g = g.map(sns.distplot,df.columns[i])
    #
    # # 热力图
    # plt.figure(figsize=(20,16))
    # C = df.corr(method="spearman")
    # mask = np.zeros_like(C,dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True # 上三角矩阵设为真
    # cmap = sns.diverging_palette(220,10,as_cmap=True) #colormap对象
    # g = sns.heatmap(C,mask=mask, cmap=cmap, square=True, annot=True,fmt='0.2f')
    # plt.show()
    # # print(C)

    # 各特征与客流量相关性及其分布
    # figure parameters
    # df_train = df['2016':'2017'].drop("weather",axis=1)
    # fcols = 2*2
    # frows = 3
    # plt.figure(figsize=(5*fcols,4*frows))
    # i=0
    # for col in df_train.columns:
    #     if i>=12:
    #         plt.figure(figsize=(5 * fcols, 4 * frows))
    #         i=0
    #     i+=1
    #     ax=plt.subplot(frows,fcols,i)
    #     sns.regplot(x=col,y='tourist',data=df_train,ax=ax,
    #                 scatter_kws={'marker':'.','s':3,'alpha':0.3},
    #                 line_kws={'color':'k'})
    #     plt.xlabel(col)
    #     plt.ylabel('tourist')
    #
    #     i+=1
    #     ax = plt.subplot(frows,fcols,i)
    #     sns.distplot(df_train[col],fit=stats.norm)

    # Threshold for removing correlated variables
    threshold = 0.1

    # Absolute value correlation matrix
    # corr_matrix = data_train1.corr().abs()
    # drop_col=corr_matrix[corr_matrix["target"]<threshold].index
    # data_all.drop(drop_col,axis=1,inplace=True)


    # #归一化
    # cols_numeric = list(df.columns)
    # cols_numeric.remove("weather")
    # def scale_minmax(col):
    #     return (col - col.min()) / (col.max() - col.min())
    #
    # scale_cols = [col for col in cols_numeric if col != 'tourist']
    # df[scale_cols] = df[scale_cols].apply(scale_minmax, axis=0)
    # df[scale_cols].describe()
    # fcols = 6
    # frows = 4
    # plt.figure(figsize=(4 * fcols, 4 * frows))
    # i = 0
    #
    # for var in cols_numeric:
    #     if i>=24:
    #         plt.figure()
    #         i=0
    #     if var != 'tourist':
    #         dat = df[[var, 'tourist']].dropna()
    #
    #         i += 1
    #         plt.subplot(frows, fcols, i)
    #         sns.distplot(dat[var], fit=stats.norm);
    #         plt.title(var + ' Original')
    #         plt.xlabel('')
    #
    #         i += 1
    #         plt.subplot(frows, fcols, i)
    #         _ = stats.probplot(dat[var], plot=plt)
    #         plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[var])))
    #         plt.xlabel('')
    #         plt.ylabel('')
    #
    #         i += 1
    #         plt.subplot(frows, fcols, i)
    #         plt.plot(dat[var], dat['tourist'], '.', alpha=0.5)
    #         plt.title('corr=' + '{:.2f}'.format(np.corrcoef(dat[var], dat['tourist'])[0][1]))
    #
    #         i += 1
    #         plt.subplot(frows, fcols, i)
    #         trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
    #         trans_var = scale_minmax(trans_var)
    #         sns.distplot(trans_var, fit=stats.norm);
    #         plt.title(var + ' Tramsformed')
    #         plt.xlabel('')
    #
    #         i += 1
    #         plt.subplot(frows, fcols, i)
    #         _ = stats.probplot(trans_var, plot=plt)
    #         plt.title('skew=' + '{:.4f}'.format(stats.skew(trans_var)))
    #         plt.xlabel('')
    #         plt.ylabel('')
    #
    #         i += 1
    #         plt.subplot(frows, fcols, i)
    #         plt.plot(trans_var, dat['tourist'], '.', alpha=0.5)
    #         plt.title('corr=' + '{:.2f}'.format(np.corrcoef(trans_var, dat['tourist'])[0][1]))

    cols_transform = df.columns[2:]
    for col in cols_transform:
        # transform column
        df.loc[:, col], _ = stats.boxcox(df.loc[:, col] + 1)
    print(df.target.describe())

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.distplot(df.target.dropna(), fit=stats.norm)
    plt.subplot(1, 2, 2)
    _ = stats.probplot(df.target.dropna(), plot=plt)

if __name__ == '__main__':
    main()