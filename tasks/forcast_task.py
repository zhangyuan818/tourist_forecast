# _*_ coding: utf-8 _*_

"""
 @version: 1.0
 @author: ZhuJingrui
 @date: 2019/8/8
 @language：Python 3.6
"""
from utils.holiday_check import IfHoliday
from utils.runtime import time_analyze
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

"""
灵山景区客流量预测，模型使用随机森林。
影响景区客流量的因素有很多，本项目主要研究天气及节假日对景区客流量的影响
"""

import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class Forecast():
    def __init__(self, *data_file):
        self.data_file = data_file    # 处理后的数据文件，可以有多个
        self.data_frame = None        # 存放数据矩阵，训练数据
        self.df_para = None           # 训练数据的均值和方差
        self.feature_cols = None      # 特征列
        self.df_train = None          # 训练矩阵，对特征项处理后
        self.model = None             # 训练好的模型，不用重复训练（每次结果不一样？）
        pass

    @time_analyze
    def _load_data(self):
        """
        加载数据
        :return:
        """
        # 将传入的数个文件依次加入训练集中
        file_list = []
        for file in self.data_file:
            df = pd.read_csv(file, encoding="utf-8").drop("scenic_area", axis=1).set_index("date") # drop:去掉“景区名称”这一列；set_index:将日期作为索引
            file_list.append(df)
        self.data_frame = pd.concat(file_list, axis=0, ignore_index=False) # 合并为1个dataframe

    def map_weather(self, weather):
        """
        处理天气函数，将天气数据转换为数字，进行量化
        :param weather:
        :return:
        """
        if "大雪" in weather:
            return 50
        elif "中雪" in weather:
            return 20
        elif "小雪" in weather:
            return 5
        elif "暴雨" in weather:
            return 95   # 自己瞎写的
        elif "大雨" in weather:
            return 75
        elif "中雨" in weather:
            return 40
        elif "阵雨" in weather:
            return 20
        elif "小雨" in weather:
            return 5
        elif "雨夹雪" in weather:
            return 5   # 自己瞎写的
        elif "阴" in weather:
            return 2
        elif "多云" in weather:
            return 0.0
        elif "晴" in weather:
            return 0.0
        else:
            return 0

    def creat_feature(self,df=pd.DataFrame()):
        """
        （1）将天气数据转化为降雨量这个可以用数字量化的指标；
        （1.5）添加特征...
        :param df:
        :return:
        """

        if df.empty:
            df=self.data_frame

        # 将天气转化为降雨量
        df["rainfall"] = df["weather"].map(self.map_weather)
        # print(set(df["天气"].values))

        #构建新的特征...

        # 1 假期天数，假期第几天
        df["num_of_holiday"] = 0
        df["ord_of_holiday"] = 0
        holiday_val = df["holiday"].values.tolist()
        holiday_val = [str(h).replace('2','1') for h in holiday_val]
        holiday_val_sta = [[k, len(list(v))] for k, v in itertools.groupby(holiday_val)]
        index=0
        for it in holiday_val_sta:
            if not it[0]=='0':
                df.loc[index:index + it[1],"num_of_holiday"] = it[1]
                df.loc[index:index + it[1],"ord_of_holiday"] = range(1,it[1]+1)
            index = index + it[1]

        # 2 昨日客流量，去年同期客流量
        df["last_year_tourist"] = None
        df["yesterday_tourist"] = None
        for i, date in enumerate(df.index.values):
            if not i==0:
                df.loc[df.index[i],"yesterday_tourist"] = df.loc[df.index[i-1],"tourist"]
            else:  # 缺失数据处理，这里直接赋值为本日客流量
                df.loc[df.index[i],"yesterday_tourist"] = df.loc[df.index[i],"tourist"]
            last_year_date = str(int(date[:4])-1)+date[4:]
            if last_year_date in df.index:
                df.loc[date,"last_year_tourist"] = df.loc[last_year_date,"tourist"]
            else:  # 缺失数据处理，这里直接赋值为本日客流量
                df.loc[date,"last_year_tourist"] = df.loc[date,"tourist"]
            # print(df.loc[i,"yesterday_tourist"],"&&&&",df.loc[date,"last_year_tourist"])
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        # # pd.set_option('max_colwidth', 100)
        # print(df[["last_year_tourist","yesterday_tourist"]])

        pass
        return df


    @time_analyze
    def process_data(self):
        """
        处理数据，包含以下几个步骤：
        特征处理放在creat_feature()函数中
        （2）提取特征项，此项目中的特征项为天气（降雨量）和节假日：工作日0、普通周末1、节假日2；
            由于节假日会对客流量造成非常明显的影响，需对这三种情况分别统计并加以分析
        (3) 数据标准化（归一化）: 它的意义是在回归分析中取消由于量纲不同、自身变异或者数值相差较大所引起的误差
        :return:
        """
        self.data_frame = self.creat_feature()[315:]

        # 指定特征项列
        self.feature_cols = ["rainfall","max_temperature","min_temperature","num_of_holiday","ord_of_holiday","last_year_tourist","yesterday_tourist"]
        # self.feature_cols = ["rainfall"]

        # 针对节假日情况分别对客流量字段进行标准化: 工作日0、普通周末1、节假日2；
        df_holiday = self.data_frame[self.data_frame["节假日"] == 2][self.feature_cols + ["tourist"]]   # 节假日
        df_weekend = self.data_frame[self.data_frame["节假日"] == 1][self.feature_cols + ["tourist"]]   # 普通周末
        df_workday = self.data_frame[self.data_frame["节假日"] == 0][self.feature_cols + ["tourist"]]   # 工作日

        # 将训练集的均值等参数存起来
        df_paralist=[[df_holiday["tourist"].mean(),df_holiday["tourist"].std()],
                     [df_weekend["tourist"].mean(),df_weekend["tourist"].std()],
                     [df_workday["tourist"].mean(),df_workday["tourist"].std()]]
        self.df_para= pd.DataFrame(data=df_paralist, columns=['mean', 'std'], index=['2', '1', '0'])

        # 数据标准化（归一化）：是数值减去均值，再除以标准差
        # mean：平均值，这一组数据的平均值, std：标准差，这一组数据的标准差
        df_holiday["tourist"] = (df_holiday["tourist"] - self.df_para.loc['2','mean'])/self.df_para.loc['2','std']
        df_weekend["tourist"] = (df_weekend["tourist"] - self.df_para.loc['1','mean'])/self.df_para.loc['1','std']
        df_workday["tourist"] = (df_workday["tourist"] - self.df_para.loc['0','mean'])/self.df_para.loc['0','std']

        # 将df_holiday、df_weekend、df_workday重新组合成一个dataFrame，并重新索引
        data_frame_eli = df_workday.append(df_weekend).append(df_holiday).reindex(self.data_frame.index)

        # 特征列标准化
        self.df_train = data_frame_eli.copy()
        self.df_train[self.feature_cols] = (self.df_train[self.feature_cols] - self.df_train[self.feature_cols].mean())/self.df_train[self.feature_cols].std()


    def random_forest(self, date, weather):
        """
        随机森林预测模型
        :param date: 预测日期,格式为："2019-08-09"
        :param weather: 预测日期当天天气,格式为："晴"、"小雨"
        :return:
        """
        features = self.df_train[self.feature_cols].as_matrix() # 输入特征项
        tourists = self.df_train["tourist"].values              # 预测结果：客流量
        x_train, x_test, y_train, y_test = train_test_split(features, tourists, test_size=0.8)
        rfr = RandomForestRegressor(n_estimators=300, n_jobs=-1, max_depth=12, max_features='sqrt') # 随机森林预测模型

        date_factor = IfHoliday().holiday_check(date) # 将日期转换为数字
        weather_factor = self.map_weather(weather)    # 将天气转化为数字
        weather_factor_norm = (weather_factor - self.df_train["rainfall"].mean())/self.df_train["rainfall"].std()
        model = rfr.fit(x_train, y_train)
        x = np.array([[weather_factor_norm]])
        tourists_pred = model.predict(x) # 预测结果，此处的是概率数字

        holiday = self.data_frame["holiday"].values
        workday_mean =  self.data_frame[ self.data_frame["holiday"]==0]["tourist"].mean()
        weekend_mean =  self.data_frame[ self.data_frame["holiday"]==1]["tourist"].mean()
        holiday_mean =  self.data_frame[ self.data_frame["holiday"]==2]["tourist"].mean()
        lambda_workday = self.data_frame[self.data_frame["holiday"]==0]["tourist"].std()
        lambda_weekend = self.data_frame[self.data_frame["holiday"]==1]["tourist"].std()
        lambda_holiday = self.data_frame[self.data_frame["holiday"]==2]["tourist"].std()

        if date_factor == 0:
            result = tourists_pred[0] * lambda_workday + workday_mean
        elif date_factor == 1:
            result = tourists_pred[0] * lambda_weekend + weekend_mean
        elif date_factor == 2:
            result = tourists_pred[0] * lambda_holiday + holiday_mean

        return int(result)

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
        x_train, x_test, y_train, y_test = train_test_split(features, tourists, test_size=0.8)
        rfr = RandomForestRegressor(n_estimators=300, n_jobs=-1, max_depth=12, max_features='sqrt')  # 随机森林预测模型
        # print(x_train)
        self.model = rfr.fit(x_train, y_train)


    def random_forest2(self, date, weather, holiday=-1):
        """
         随机森林模型预测部分
        :param date: 预测日期,格式为："2019-08-09"
        :param weather: 预测日期当天天气,格式为："晴"、"小雨"
        :param holiday:
        :return:
        """
        # 预测部份
        if holiday == -1:
            date_factor = IfHoliday().holiday_check(date)  # 将日期转换为数字
        else:
            date_factor = holiday
        weather_factor = self.map_weather(weather)  # 将天气转化为数字
        weather_factor_norm = (weather_factor - self.df_train["rainfall"].mean()) / self.df_train["rainfall"].std()
        x = np.array([[weather_factor_norm]])
        tourists_pred = self.model.predict(x) # 预测结果，此处的是概率数字

        if date_factor == 0:
            result = tourists_pred[0] * self.df_para.loc['0','std'] + self.df_para.loc['0','mean']
        elif date_factor == 1:
            result = tourists_pred[0] * self.df_para.loc['1','std'] + self.df_para.loc['1','mean']
        elif date_factor == 2:
            result = tourists_pred[0] * self.df_para.loc['2','std'] + self.df_para.loc['2','mean']

        return int(result)

    @time_analyze
    def predition(self, predition_data):
        """
        预测
        :param predition_data: DataFrame类型,包括“日期”，“天气” 等特征列
        :return: series类型，预测结果
        """
        # 构建没有的特征
        predition_data = self.creat_feature(predition_data)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        # pd.set_option('max_colwidth', 100)
        print(predition_data[["tourist","last_year_tourist","yesterday_tourist"]])

        factor_norm = (predition_data[self.feature_cols] - self.df_train[self.feature_cols].mean()) / self.df_train[self.feature_cols].std()
        # print(factor_norm.shape)
        # print(type(factor_norm))
        # print(factor_norm)
        x = np.array(factor_norm)

        tourists_pred = self.model.predict(x) # 预测结果，此处的是概率数字

        pd_holiday_i = predition_data["holiday"] == 2  # 节假日
        pd_weekend_i = predition_data["holiday"] == 1  # 普通周末
        pd_workday_i = predition_data["holiday"] == 0  # 工作日

        result= pd.Series(index=predition_data.index, name='predicted_value')
        result[pd_holiday_i] = tourists_pred[pd_holiday_i] * self.df_para.loc['2','std'] + self.df_para.loc['2','mean']
        result[pd_weekend_i] = tourists_pred[pd_weekend_i] * self.df_para.loc['1','std'] + self.df_para.loc['1','mean']
        result[pd_workday_i] = tourists_pred[pd_workday_i] * self.df_para.loc['0','std'] + self.df_para.loc['0','mean']

        result[:] = np.trunc(result[:].values).astype(int)  # 将结果序列中的数取整

        return result

    def analysis(self, predict, real, date):
        """
        画预测数据和真实数据图
        :param predict:ndarray 
        :param real: ndarray
        :param date: nadarray
        :return: 
        """
        #参数输出
        r2 = r2_score(real, predict)  # R2：决定系数（拟合优度）
        rmse = np.sqrt(mean_squared_error(real, predict))  # 平均平方误差（均方差）
        mae = mean_absolute_error(real, predict)  # 平均绝对误差
        print("r2: %s" % r2)
        print("rmse: %s" % rmse)
        print("mae: %s" % mae)

        # 画图部分
        year = date[0][:4]
        month = []
        for index, day in enumerate(date):
            if day[-2:] == '01':
                month.append([index, day[5:7]])
        month = [[row[i] for row in month] for i in range(len(month[0]))]

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
        title = year + " line chart of ground truth and predicted value"
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
    def run(self,date, weather):
        self._load_data()
        self.process_data()
        return self.random_forest(date, weather)

    @time_analyze
    def run2(self,predition_data):
        """
        训练模型并预测
        :param predition_data: DataFrame类型,包括“日期”，“天气” 等特征列
        :return: series类型，预测结果
        """
        if self.model == None:
            self._load_data()
            self.process_data()
            print("train model...")
            self.train_model()
        return self.predition(predition_data)

if __name__ == "__main__":
    train_file1 = "../data/2015data.csv"
    train_file2 = "../data/2016data.csv"
    train_file3 = "../data/2017data.csv"
    train_file4 = "../data/2018data.csv"
    train_file5 = "../data/2019data.csv"
    #date = "2018-03-04"
    #weather = "多云"

    forecast = Forecast(train_file1,train_file2,train_file3,train_file4)

    predict_df = pd.read_csv(train_file5, encoding="utf-8").drop("scenic_area", axis=1).set_index("date")
    # forecast.run2(predict_df)
    predict_result = forecast.run2(predict_df)
    forecast.analysis(predict_result[:].values, predict_df["tourist"].values, predict_df.index.values)
    # print(pre2018)
    '''
    result = Forecast(file_out).run(date, weather)
    print(result)'''
    # print(forecast.map_weather("大到暴雨"))

    pass