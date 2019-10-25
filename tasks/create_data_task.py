# _*_ coding: utf-8 _*_

"""
 @version: 1.0
 @author: ZhangYuan
 @date: 2019/10/16
 @language：Python 3.7
"""

import codecs


"""
构造数据
"""

from utils.reader import Reader
from utils.holiday_check import IfHoliday
from datetime import datetime
import os

class DataCreator():
    WEATHER_FILE = "../data/weather_data/YYYYweather.txt"
    TOURIST_FILE = "../data/tourist_data/YYYYtourist.xlsx"
    WEATHER_OUT = "../data/weather/YYYYweather.csv"
    HOLIDAY_OUT = "../data/holiday/YYYYholiday.json"
    FILE_OUT = "../data/YYYYdata.csv"
    WEATHER_FILE2 = "../data/weather_data/2016-2019weather.xlsx"
    def __init__(self):
        pass

    # def create_data(self, weather_file, tourist_file, file_out):
    #     """
    #     构造数据集
    #     :param weather_file: 天气数据文件
    #     :param tourist_file: 客流量数据文件
    #     :param file_out: 输出文件
    #     :return:
    #     """
    #     weather_dict = Reader().read_weather(weather_file)##这个方法用不了了
    #     tourist_dict = Reader().read_tourist(tourist_file)
    #     with codecs.open(file_out,'a+','utf-8') as fout:
    #         fout.write("景区名称,日期,客流量,天气,节假日" + "\n")
    #         for date, tourist in tourist_dict.items():
    #             name = "上饶灵山景区"  #景区名称
    #             weather = weather_dict[date]
    #             try:
    #                 is_holiday = IfHoliday().holiday_check(date)
    #             except ConnectionError:
    #                 time.sleep(10)
    #                 is_holiday = IfHoliday().holiday_check(date)
    #             text = name + "," + date + "," + str(tourist) + "," + weather + "," + str(is_holiday) + "\n"
    #             fout.write(text)

    def create_data2(self, year, tourist_file=TOURIST_FILE, weather_file=WEATHER_FILE,weather_file2=WEATHER_FILE2):
        """
        输入年份会根据tourist_file和weather_file中的数据构造数据集
        :param year: 数据年份,数值类型
        :param tourist_file: 客流量数据文件，默认路径TOURIST_FILE
        :param weather_file:  天气数据文件，默认路径WEATHER_FILE
        :param weather_file2: 天气详细数据文件，默认路径WEATHER_FILE2
        :return:
        """
        tourist_file = tourist_file.replace('YYYY',str(year))
        weather_file = weather_file.replace('YYYY',str(year))
        weather_out = self.WEATHER_OUT.replace('YYYY',str(year))
        holiday_out = self.HOLIDAY_OUT.replace('YYYY',str(year))
        file_out = self.FILE_OUT.replace('YYYY',str(year))


        if not os.path.isfile(weather_out):
            Reader().get_weather(weather_file, weather_out)
        weather_dict = Reader().read_weather(weather_out)
        weather_dict2 = Reader().read_weather2(weather_file2)

        # 从api获得year的节假日数据存入holiday_out中，避免多次http请求
        if not os.path.isfile(holiday_out):
            IfHoliday().get_year_holiday(year, holiday_out)
        holiday_dict = Reader().read_holiday(holiday_out)

        tourist_dict = Reader().read_tourist(tourist_file) #用字典可能会导致日期顺序不对，影响特征工程正确率


        with codecs.open(file_out,'a+','utf-8') as fout:
            fout.write("scenic_area,date,tourist,holiday,weather,min_temperature,max_temperature,mean_temperature,"+
                       "humidity,wind_speed,precipitation,cloudage" + "\n")
            for date, tourist in tourist_dict.items():
                name = "上饶灵山景区"  #景区名称

                weather = weather_dict[date][0]
                # max_temperature = weather_dict[date][1]
                # min_temperature = weather_dict[date][2]
                if date in weather_dict2.keys():
                    weather_info = str(weather_dict2[date]).strip('[]')
                else:
                    weather_info = str("None,"*7).strip('[],')
                '''下面这部分代码可以考虑放进IfHoliday()中'''
                holiday = 0
                day = datetime.strptime(date.replace('-', ''), "%Y%m%d").date()
                if date in holiday_dict.keys():
                    if holiday_dict[date]:
                        holiday = 2
                elif day.weekday() in [5, 6]:
                    holiday = 1

                text = name + "," + date + "," + str(tourist) + "," + str(holiday) + "," + weather + "," + weather_info + "\n"
                fout.write(text)



if __name__ == "__main__":
    # weather_file = "../data/2015weather.csv"
    # tourist_file = "../data/2015年灵山景区旅游人数统计汇总表.xlsx"
    # file_out = "../data/2015data.csv"
    # DataCreator().create_data(weather_file, tourist_file, file_out)
    #holiday_out = "../data/2015holiday.json"
    for year in range(2015,2020):
        DataCreator().create_data2(year)
    pass