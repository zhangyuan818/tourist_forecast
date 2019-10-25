#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
 @version: 1.0
 @author: ZhangYuan
 @date: 2019/10/16
 @language：Python 3.7
"""

import matplotlib.pyplot as plt
import pandas as pd
import itertools
from scipy import interpolate
from scipy import stats
import numpy as np
import seaborn as sns


class FeatureEngineering():
    def __init__(self, *data_file):
        self.data_file = data_file    # 处理后的数据文件，可以有多个
        self.data_frame = None        # 存放数据矩阵，训练数据
        self._load_data()
        pass

    def _load_data(self):
        """
        加载数据
        :return:
        """
        # 将传入的数个文件依次加入训练集中
        file_list = []
        for file in self.data_file:
            df = pd.read_csv(file, encoding="utf-8",na_values=["None"," ''"]).drop("scenic_area", axis=1).set_index("date") # drop:去掉“景区名称”这一列；set_index:将日期作为索引
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

    def interpolation(self, feature,type=0):
        """
        插值法填补缺失值,type:temperature=0; tourist=1; humidity,cloudage=2; wind_speed=3
        :param feature:
        :param type: 0:float; 1:int>=0; 2:int 0~100; 3:float>=0
        :return:
        """
        realdata = self.data_frame[feature]
        realindex = np.arange( 1, len(realdata) + 1)
        index = realdata.notnull()
        nullindex = realdata.isnull()
        y=realdata[index]
        x = realindex[index.values]
        f = interpolate.interp1d(x, y, kind="quadratic")
        ynew = f(realindex)
        ynew[index.values] = y
        if type==1:
            ynew[ynew < 0] = 0
            ynew = np.floor(ynew) #向下取整
        elif type==2:
            ynew[ynew < 0] = 0
            ynew[ynew > 100] = 100
            ynew = np.floor(ynew) #向下取整
        elif type==3:
            ynew[ynew < 0] = 0
        self.data_frame[feature] = ynew

    def creat_feature(self):
        """
        添加特征...
        :param:
        :return:
        """
        self.data_frame = self.data_frame[56:]  # 15年4.15日之前没收门票，客流量不准，去掉

        # 填补客流量空缺
        self.interpolation("tourist",1)

        # 将天气转化为降雨量
        # self.data_frame["rainfall"] = self.data_frame["weather"].map(self.map_weather)

        # 构建新的特征...

        # 1 假期天数，假期第几天
        self.data_frame["num_of_holiday"] = 0
        self.data_frame["ord_of_holiday"] = 0
        holiday_val = self.data_frame["holiday"].values.tolist()
        holiday_val = [str(h).replace('2', '1') for h in holiday_val]
        holiday_val_sta = [[k, len(list(v))] for k, v in itertools.groupby(holiday_val)]
        index = 0
        for it in holiday_val_sta:
            if not it[0] == '0':
                self.data_frame.loc[index:index + it[1], "num_of_holiday"] = it[1]
                self.data_frame.loc[index:index + it[1], "ord_of_holiday"] = range(1, it[1] + 1)
            index = index + it[1]


        # 2 昨日客流量，去年同期客流量
        self.data_frame["last_year_tourist"]=np.nan
        self.data_frame["yesterday_tourist"]=np.nan

        for i, date in enumerate(self.data_frame.index.values):
            if not i == 0:
                yeaterday = self.data_frame.index[i - 1]
                self.data_frame.loc[date, "yesterday_tourist"] = self.data_frame.loc[yeaterday, "tourist"]
            else:  # 缺失数据处理，这里直接赋值为本日客流量
                self.data_frame.loc[date, "yesterday_tourist"] = self.data_frame.loc[date, "tourist"]
            last_year_date = str(int(date[:4]) - 1) + date[4:]
            previous_year_date = str(int(date[:4]) - 2) + date[4:]
            if last_year_date in self.data_frame.index:
                self.data_frame.loc[date, "last_year_tourist"] = self.data_frame.loc[last_year_date, "tourist"]
            elif previous_year_date in self.data_frame.index: #去年数据缺失就赋值为前年客流
                self.data_frame.loc[date, "last_year_tourist"] = self.data_frame.loc[previous_year_date, "tourist"]
            # else:  # 缺失数据处理，这里直接赋值为本日客流量
            #     self.data_frame.loc[date, "last_year_tourist"] = self.data_frame.loc[date, "tourist"]

        self.data_frame = self.data_frame[363:] #去掉2015年
        self.interpolation("last_year_tourist",1)  # 闰年或数据缺失导致没有上年同期
        self.interpolation("min_temperature")
        self.interpolation("max_temperature")
        self.interpolation("mean_temperature")
        self.interpolation("cloudage",2)
        self.interpolation("humidity",2)
        self.interpolation("wind_speed",3)
        self.interpolation("precipitation",3)

        # 3 人体舒适度
        self.data_frame["comfort_index"] = np.nan
        T = self.data_frame["mean_temperature"]
        U = self.data_frame["humidity"]
        V = self.data_frame["wind_speed"]
        self.data_frame["comfort_index"] = 1.8*T+32-0.55*(1.8*T-26)*(1-U/100)-3.2*np.sqrt(V)
        pass

    def save_feature_data(self, feature_data_file):
        """
        将创建了新的特征的数据存起来
        :return:
        """
        self.data_frame.to_csv(feature_data_file, sep=',', header=True, index=True)

    def year_data_chart(self, year1_data, year2_data=pd.DataFrame()):
        """
        此方法用来画1-2年的客流量图，待调整，暂时不用
        :param year1_data:
        :param year2_data:
        :return:
        """
        x1 = range(1, len(year1_data) + 1, 1)


        date1 = year1_data['date'].values
        title1 = "ground truth for " + date1[0][:4]
        month1 = []
        for index, day in enumerate(date1):
            if day[-2:] == '01':
                month1.append([index, day[5:7]])
        month1 = [[row[i] for row in month1] for i in range(len(month1[0]))]

        if year2_data.empty:
            plt.figure(figsize=(20, 5))

            plt.title(title1)
            plt.plot(x1, year1_data['tourist'].values)
            plt.xticks(month1[0], month1[1])
            plt.xlabel("month")

        else:
            x2 = range(1, len(year2_data) + 1, 1)
            date2 = year2_data['date'].values
            title2 = "ground truth for " + date2[0][:4]
            month2 = []
            for index, day in enumerate(date2):
                if day[-2:] == '01':
                    month2.append([index, day[5:7]])
            month2 = [[row[i] for row in month2] for i in range(len(month2[0]))]
            # 画图
            plt.figure(figsize=(20, 10))
            ax2 = plt.subplot(2, 1, 2)
            ax1 = plt.subplot(2, 1, 1)

            # ###绘制真实值与预测值比较折线图

            plt.sca(ax1)
            plt.title(title1)
            plt.plot(x1, year1_data['tourist'].values)
            plt.xticks(month1[0], month1[1])
            # plt.xlabel("month")
            # plt.setp(ax1.get_xticklabels(), visible=False)

            plt.sca(ax2)
            plt.title(title2)
            plt.plot(x2, year2_data['tourist'].values)
            plt.xticks(month2[0], month2[1])
            plt.xlabel("month")

def data_analysis(df):
    """
    分析数据相关性等并画图
    :return:
    """

    df.head()
    # for column in
    C = df.corr()
    print(C)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=-90)
    # sns.pairplot(df)

    # sns.pairplot(df, hue = "holiday")
    sns.heatmap(C) #热度图
    # g = sns.PairGrid(df)
    # g.map_diag(sns.distplot)
    # g.map_upper(plt.scatter)
    # g.map_lower(sns.kdeplot)
    plt.show()
    pass

def feature_enginneering():
    train_file1 = "../data/2015data.csv"
    train_file2 = "../data/2016data.csv"
    train_file3 = "../data/2017data.csv"
    train_file4 = "../data/2018data.csv"
    train_file5 = "../data/2019data.csv"
    feature_engineering = FeatureEngineering(train_file1, train_file2, train_file3, train_file4, train_file5)
    feature_engineering.creat_feature()
    feature_data_file = "../data/featureData.csv"
    feature_engineering.save_feature_data(feature_data_file)


if __name__=="__main__":
    # feature_enginneering()
    pass

    # feature_engineering.data_analysis()

    # feature_engineering.year_data_chart(data2015, data2016)
    # feature_engineering.year_data_chart(data2017, data2018)
    # feature_engineering.year_data_chart(data2019)
    # plt.show()