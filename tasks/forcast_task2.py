# _*_ coding: utf-8 _*_

"""
 @version: 1.0
 @author: ZhangYuan
 @date: 2019/10/16
 @language：Python 3.7
"""

from utils.runtime import time_analyze
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

"""
灵山景区客流量预测，模型使用随机森林。
影响景区客流量的因素有很多，本项目主要研究天气及节假日对景区客流量的影响
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class Forecast():
    def __init__(self, data_file):
        self.data_file = data_file    # 处理后的数据文件
        self.data_frame = None        # 存放数据矩阵，训练数据
        self.df_para = {}           # 训练数据的均值和方差
        self.feature_cols = None      # 特征列
        self.df_train = None          # 训练矩阵，对特征项处理后,归一化后
        self.model = None             # 训练好的模型，不用重复训练（每次结果不一样？）
        self.prediction_df = None
        pass

    @time_analyze
    def _load_data(self):
        """
        加载数据
        :return:
        """
        df = pd.read_csv(self.data_file, encoding="utf-8")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date") # set_index:将日期作为索引
        # self.data_frame = df.truncate(before = '2016', after='2019') # 删掉2016(不含)以前，2019(含)以后的数据
        self.data_frame = df.truncate(after='2018')
        self.prediction_df = df['2018']



    @time_analyze
    def process_data(self):
        """
        处理数据
        数据标准化（归一化）: 它的意义是在回归分析中取消由于量纲不同、自身变异或者数值相差较大所引起的误差
        :return:
        """

        # 指定特征项列
        self.feature_cols = ["holiday","num_of_holiday","ord_of_holiday","last_year_tourist","yesterday_tourist",
                             "max_temperature", "comfort_index", "precipitation", "cloudage"]
        # self.feature_cols = ["holiday","num_of_holiday","ord_of_holiday","last_year_tourist","yesterday_tourist",
        #                      "max_temperature","min_temperature","mean_temperature","humidity","wind_speed",
        #                      "comfort_index","precipitation","cloudage"]

        self.df_train = self.data_frame[self.feature_cols + ["tourist"]].reindex(self.data_frame.index)
        self.df_train["tourist"] = (self.df_train["tourist"] - self.df_train["tourist"].mean())/self.df_train["tourist"].std()

        # 特征列标准化
        self.df_train[self.feature_cols] = (self.df_train[self.feature_cols] - self.df_train[self.feature_cols].mean())/self.df_train[self.feature_cols].std()


    @time_analyze
    def train_model(self):
        """
        随机森林预测模型的建立
        :return:
        """
        # 建模部分
        features = self.df_train[self.feature_cols].values  # 输入特征项
        # print(features)
        tourists = self.df_train["tourist"].values  # 预测结果：客流量
        x_train, x_test, y_train, y_test = train_test_split(features, tourists, test_size=0.3)
        rfr = RandomForestRegressor(n_estimators=300, n_jobs=-1)  # 随机森林预测模型
        # print(x_train)
        self.model = rfr.fit(x_train, y_train)

    @time_analyze
    def prediction(self):
        """
        预测
        :param
        :return: series类型，预测结果
        """
        prediction_data = self.prediction_df
        factor_norm = (prediction_data[self.feature_cols] - self.data_frame[self.feature_cols].mean()) / self.data_frame[self.feature_cols].std()
        x = np.array(factor_norm)

        tourists_pred = self.model.predict(x) # 预测结果，此处的是概率数字

        result= pd.Series(index=prediction_data.index, name='predicted_value')
        result[:] = tourists_pred * self.data_frame["tourist"].std() + self.data_frame["tourist"].mean()

        result[:] = np.trunc(result[:].values).astype(int)  # 将结果序列中的数取整
        return result

    def analysis(self,predict):
        """
        画预测数据和真实数据图
        :param predict:
        :return:
        """
        predict = predict[:].values
        real = self.prediction_df["tourist"].values
        date = self.prediction_df.index
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

    @time_analyze
    def run2(self):
        """
        训练模型并预测
        :param
        :return: series类型，预测结果
        """
        if self.model == None:
            self._load_data()
            self.process_data()
            print("train model...")
            self.train_model()
        return self.prediction()

if __name__ == "__main__":

    forecast = Forecast("../data/featureData.csv")

    # predict_df = pd.read_csv(train_file5, encoding="utf-8").drop("scenic_area", axis=1).set_index("date")
    # forecast.run2(predict_df)
    predict_result = forecast.run2()
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(suppress=True)

    forecast.analysis(predict_result)


    pass