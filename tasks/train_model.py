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
import warnings
warnings.filterwarnings("ignore")
# modelling
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score,cross_val_predict,KFold
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)

    df = pd.read_csv("../data/featureData.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.drop(["weather"], axis=1)
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
    # threshold = 0.1

    # Absolute value correlation matrix
    # corr_matrix = data_train1.corr().abs()
    # drop_col=corr_matrix[corr_matrix["target"]<threshold].index
    # df.drop(drop_col,axis=1,inplace=True)

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

    # 能不能用两说
    # cols_transform = df.columns[2:]
    # for col in cols_transform:
    #     # transform column
    #     df.loc[:, col], _ = stats.boxcox(df.loc[:, col] + 1)
    # print(df.target.describe())
    #
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # sns.distplot(df.target.dropna(), fit=stats.norm)
    # plt.subplot(1, 2, 2)
    # _ = stats.probplot(df.target.dropna(), plot=plt)

    # 处理节假日特征
    df_holiday = pd.get_dummies(df['holiday'],prefix='holiday')
    df = pd.concat((df,df_holiday),axis=1)
    df.drop(['holiday'],axis=1,inplace=True)
    # 分训练集和测试集
    df_train = df.truncate(after='2018')
    df_test = df['2018']

    # rkfold = RepeatedKFold(n_splits=5, n_repeats=5)

    # feature_cols = ["holiday", "num_of_holiday", "ord_of_holiday", "last_year_tourist", "yesterday_tourist",
    #                 "max_temperature", "min_temperature", "mean_temperature", "humidity", "wind_speed",
    #                 "comfort_index", "precipitation", "cloudage"]

    # 分特征
    train_y = df_train["tourist"]
    train_X = df_train.drop(['tourist'],axis=1)
    test_y = df_test['tourist']
    test_X = df_test.drop(['tourist'],axis=1)
    # 标准化
    scaler_X = StandardScaler().fit(train_X)
    train_X = scaler_X.transform(train_X)
    test_X = scaler_X.transform(test_X)

    scaler_y = StandardScaler().fit(np.array(train_y).reshape(-1,1))
    train_y = scaler_y.transform(np.array(train_y).reshape(-1,1))
    # 测试数据的tourist列不用标准化

    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)
    # ---------------------------------------------------------
    # function to get training samples
    # def get_feature_data(data):
    #     y = data["tourist"]
    #     X = data[feature_cols]
    #     # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=100)
    #     return X, y


    # metric for evaluation
    def rmse(y_true, y_pred):
        diff = y_pred - y_true
        sum_sq = sum(diff ** 2)
        n = len(y_pred)

        return np.sqrt(sum_sq / n)


    def mse(y_ture, y_pred):
        return mean_squared_error(y_ture, y_pred)


    # scorer to be used in sklearn model fitting
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    mse_scorer = make_scorer(mse, greater_is_better=False)


    # function to detect outliers based on the predictions of a model
    # def find_outliers(model, X, y, sigma=3):
    #
    #     # predict y values using model
    #     try:
    #         y_pred = pd.Series(model.predict(X), index=y.index)
    #     # if predicting fails, try fitting the model first
    #     except:
    #         model.fit(X, y)
    #         y_pred = pd.Series(model.predict(X), index=y.index)
    #
    #     # calculate residuals between the model prediction and true y values
    #     resid = y - y_pred
    #     mean_resid = resid.mean()
    #     std_resid = resid.std()
    #
    #     # calculate z statistic, define outliers to be where |z|>sigma
    #     z = (resid - mean_resid) / std_resid
    #     outliers = z[abs(z) > sigma].index
    #
    #     # print and plot the results
    #     print('R2=', model.score(X, y))
    #     print('rmse=', rmse(y, y_pred))
    #     print("mse=", mean_squared_error(y, y_pred))
    #     print('---------------------------------------')
    #
    #     print('mean of residuals:', mean_resid)
    #     print('std of residuals:', std_resid)
    #     print('---------------------------------------')
    #
    #     print(len(outliers), 'outliers:')
    #     print(outliers.tolist())
    #
    #     plt.figure(figsize=(15, 5))
    #     ax_131 = plt.subplot(1, 3, 1)
    #     plt.plot(y, y_pred, '.')
    #     plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    #     plt.legend(['Accepted', 'Outlier'])
    #     plt.xlabel('y')
    #     plt.ylabel('y_pred')
    #
    #     ax_132 = plt.subplot(1, 3, 2)
    #     plt.plot(y, y - y_pred, '.')
    #     plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    #     plt.legend(['Accepted', 'Outlier'])
    #     plt.xlabel('y')
    #     plt.ylabel('y - y_pred')
    #
    #     ax_133 = plt.subplot(1, 3, 3)
    #     z.plot.hist(bins=50, ax=ax_133)
    #     z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    #     plt.legend(['Accepted', 'Outlier'])
    #     plt.xlabel('z')
    #
    #     plt.savefig('outliers.png')
    #
    #     return outliers

    # X_train, X_valid, y_train, y_valid = get_training_data()
    # test = get_test_data()

    # find and remove outliers using a Ridge model
    # outliers = find_outliers(Ridge(), X_train, y_train)

    # permanently remove these outliers from the data
    # df_train = data_all[data_all["oringin"]=="train"]
    # df_train["label"]=data_train.target1
    # df_train=df_train.drop(outliers)
    # X_outliers = X_train.loc[outliers]
    # y_outliers = y_train.loc[outliers]
    # X_t = X_train.drop(outliers)
    # y_t = y_train.drop(outliers)
    #

    # def get_trainning_data_omitoutliers():
    #     y1 = y_t.copy()
    #     X1 = X_t.copy()
    #     return X1, y1

    def train_model(model, param_grid=[], X=[], y=[],
                    splits=5, repeats=5):

        # get unmodified training data, unless data to use already specified
        if len(y) == 0:
            X = train_X
            y = train_y
            # poly_trans=PolynomialFeatures(degree=2)
            # X=poly_trans.fit_transform(X)
            # X=MinMaxScaler().fit_transform(X)

        # create cross-validation method
        rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)

        # perform a grid search if param_grid given
        if len(param_grid) > 0:
            # setup grid search parameters
            gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                                   scoring="neg_mean_squared_error",
                                   verbose=1, return_train_score=True)

            # search the grid
            gsearch.fit(X, y)

            # extract best model from the grid
            model = gsearch.best_estimator_
            best_idx = gsearch.best_index_

            # get cv-scores for best model
            grid_results = pd.DataFrame(gsearch.cv_results_)
            cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
            cv_std = grid_results.loc[best_idx, 'std_test_score']

        # no grid search, just cross-val score for given model
        else:
            grid_results = []
            cv_results = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=rkfold)
            cv_mean = abs(np.mean(cv_results))
            cv_std = np.std(cv_results)

        # combine mean and std cv-score in to a pandas series
        cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

        # predict y using the fitted model
        y_pred = model.predict(X)

        # # print stats on model performance
        # print('----------------------')
        # print(model)
        # print('score=', model.score(X, y))
        # print('rmse=', rmse(y, y_pred))
        # print('mse=', mse(y, y_pred))
        # print('cross_val: mean=', cv_mean, ', std=', cv_std)
        # print('----------------------')
        # residual plots
        # y_pred = pd.Series(y_pred, index=y.index)
        # resid = y - y_pred
        # mean_resid = resid.mean()
        # std_resid = resid.std()
        # z = (resid - mean_resid) / std_resid
        # n_outliers = sum(abs(z) > 3)
        #
        # plt.figure(figsize=(15, 5))
        # ax_131 = plt.subplot(1, 3, 1)
        # plt.plot(y, y_pred, '.')
        # plt.xlabel('y')
        # plt.ylabel('y_pred')
        # plt.title('corr = {:.3f}'.format(np.corrcoef(y, y_pred)[0][1]))
        # ax_132 = plt.subplot(1, 3, 2)
        # plt.plot(y, y - y_pred, '.')
        # plt.xlabel('y')
        # plt.ylabel('y - y_pred')
        # plt.title('std resid = {:.3f}'.format(std_resid))
        #
        # ax_133 = plt.subplot(1, 3, 3)
        # z.plot.hist(bins=50, ax=ax_133)
        # plt.xlabel('z')
        # plt.title('{:.0f} samples with z>3'.format(n_outliers))

        return model, cv_score, grid_results


    # places to store optimal models and scores
    opt_models = dict()
    score_models = pd.DataFrame(columns=['mean', 'std'])

    # no. k-fold splits
    splits = 5
    # no. k-fold iterations
    repeats = 5

    # ---------------------------------------------------------
    model = 'Ridge'

    opt_models[model] = Ridge()
    alph_range = np.arange(0.25, 6, 0.25)
    param_grid = {'alpha': alph_range}

    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                            splits=splits, repeats=repeats)

    cv_score.name = model
    score_models = score_models.append(cv_score)

    plt.figure()
    plt.errorbar(alph_range, abs(grid_results['mean_test_score']),
                 abs(grid_results['std_test_score']) / np.sqrt(splits * repeats))
    plt.xlabel('alpha')
    plt.ylabel('score')

    # ---------------------------------------------------------
    model = 'Lasso'

    opt_models[model] = Lasso()
    alph_range = np.arange(1e-4, 1e-3, 4e-5)
    param_grid = {'alpha': alph_range}

    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                            splits=splits, repeats=repeats)

    cv_score.name = model
    score_models = score_models.append(cv_score)

    plt.figure()
    plt.errorbar(alph_range, abs(grid_results['mean_test_score']),
                 abs(grid_results['std_test_score']) / np.sqrt(splits * repeats))
    plt.xlabel('alpha')
    plt.ylabel('score')
    # ---------------------------------------------------------
    model = 'ElasticNet'
    opt_models[model] = ElasticNet()

    param_grid = {'alpha': np.arange(1e-4, 1e-3, 1e-4),
                  'l1_ratio': np.arange(0.1, 1.0, 0.1),
                  'max_iter': [100000]}

    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                            splits=splits, repeats=1)

    cv_score.name = model
    score_models = score_models.append(cv_score)
    # ---------------------------------------------------------
    model = 'SVR'
    opt_models[model] = SVR()

    crange = np.arange(0.1, 1.0, 0.1)
    param_grid = {'C': crange,
                  'max_iter': [1000]}

    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                            splits=splits, repeats=repeats)

    cv_score.name = model
    score_models = score_models.append(cv_score)

    plt.figure()
    plt.errorbar(crange, abs(grid_results['mean_test_score']),
                 abs(grid_results['std_test_score']) / np.sqrt(splits * repeats))
    plt.xlabel('C')
    plt.ylabel('score')
    # ---------------------------------------------------------
    model = 'KNeighbors'
    opt_models[model] = KNeighborsRegressor()

    param_grid = {'n_neighbors': np.arange(3, 11, 1)}

    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                            splits=splits, repeats=1)

    cv_score.name = model
    score_models = score_models.append(cv_score)

    plt.figure()
    plt.errorbar(np.arange(3, 11, 1), abs(grid_results['mean_test_score']),
                 abs(grid_results['std_test_score']) / np.sqrt(splits * 1))
    plt.xlabel('n_neighbors')
    plt.ylabel('score')
    # ----------------------------------------------------------
    model = 'GradientBoosting'
    opt_models[model] = GradientBoostingRegressor()

    param_grid = {'n_estimators': [150, 250, 350],
                  'max_depth': [1, 2, 3],
                  'min_samples_split': [5, 6, 7]}

    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                            splits=splits, repeats=1)

    cv_score.name = model
    score_models = score_models.append(cv_score)
    # ------------------------------------------------------------
    model = 'XGB'
    opt_models[model] = XGBRegressor()

    param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                  'max_depth': [1, 2, 3],
                  }

    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                            splits=splits, repeats=1)

    cv_score.name = model
    score_models = score_models.append(cv_score)

    # ---------------------------------------------------------
    model = 'RandomForest'
    opt_models[model] = RandomForestRegressor()

    param_grid = {'n_estimators': [100, 150, 200],
                  'max_features': [1, 2, 3, 4, 5],
                  'min_samples_split': [2, 4, 6]}

    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,
                                                            splits=5, repeats=1)

    cv_score.name = model
    score_models = score_models.append(cv_score)

    def analysis(predict,real_y):
        """
        画预测数据和真实数据图
        """
        predict = predict.reshape(-1)
        real = real_y.values
        date = real_y.index
        # date = date_dt.apply(lambda x: datetime.strftime(x, "%d-%m-%Y"))

        #参数输出
        r2 = r2_score(real, predict)  # R2：决定系数（拟合优度）
        rmse = np.sqrt(mean_squared_error(real, predict))  # 平均平方误差（均方差）
        mae = mean_absolute_error(real, predict)  # 平均绝对误差
        print("r2: %s" % r2)
        print("rmse: %s" % rmse)
        print("mae: %s" % mae)

        # 画图部分
        year = date.year
        month = []
        for index, day in enumerate(date):
            if day.day == 1:
                month.append([index, str(day.month)])
        month = [[row[i] for row in month] for i in range(len(month[0]))] # 转置

        x=range(1,len(predict)+1,1)
        # print("predict type:",type(predict),"real type:",type(real))
        lower_err = real - predict
        upper_err = predict - real
        lower_err[lower_err<0] = 0
        upper_err[upper_err<0] = 0
        err = [lower_err,upper_err]
        # print(err)

        # 画图
        plt.figure(figsize=(20, 10))
        ax2 = plt.subplot(2, 1, 2)
        ax1 = plt.subplot(2, 1, 1,sharex=ax2)

        # ###绘制真实值与预测值比较折线图
        title = str(year[0]) + " line chart of ground truth and predicted value"
        plt.sca(ax1)
        plt.title(title)
        plt.plot(x, real)
        plt.plot(x, predict, color="red")
        plt.legend(["True Ground", "Prediction"])
        plt.setp(ax1.get_xticklabels(), visible=False)


        plt.sca(ax2)
        plt.title("Error graph of predicted value")
        plt.errorbar(x,np.zeros_like(x),yerr=err)
        plt.xticks(month[0],month[1])
        plt.xlabel("month")
        plt.show()
    def model_predict(test_data, test_y=None):
        # poly_trans=PolynomialFeatures(degree=2)
        # test_data1=poly_trans.fit_transform(test_data)
        # test_data=MinMaxScaler().fit_transform(test_data)
        i = 0
        y_predict_total = np.zeros((test_data.shape[0],1))
        for model in opt_models.keys():
            if model != "LinearSVR" and model != "KNeighbors":
                y_predict = opt_models[model].predict(test_data)
                y_predict = y_predict.reshape((-1,1))
                y_predict_total += y_predict
                i += 1
            if len(test_y) > 0:
                ini_pre_y = scaler_y.inverse_transform(y_predict)
                print("{}_mse:".format(model), mean_squared_error(ini_pre_y, test_y))
                analysis(ini_pre_y, test_y)
        y_predict_mean = np.round(y_predict_total / i, 3)
        if len(test_y) > 0:
            print("mean_mse:", mean_squared_error(y_predict_mean, test_y))
        else:
            y_predict_mean = pd.Series(y_predict_mean)
            return y_predict_mean


    model_predict(test_X, test_y)