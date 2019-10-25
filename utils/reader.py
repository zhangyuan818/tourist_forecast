# _*_ coding: utf-8 _*_

"""
 @version: 1.0
 @author: ZhuJingrui
 @date: 2019/1/28
 @language：Python 3.6
"""
import codecs
import json
import datetime
import xlrd
from xlrd import xldate_as_tuple



class Reader():
    def __init__(self):
        pass

    def get_weather(self, file_in, file_out):
        """
        将天气网站的数据，截取其中日期,白天天气,最高温度，最低温度/天气文件的预处理
        :param file_in: 输入文件
        :param file_out 输出文件
        :return:
        """
        with codecs.open(file_out,'a+','utf-8') as fout:
            with codecs.open(file_in,'r','utf-8') as fin:
                for line in fin.readlines():
                    date_weather = line.strip().split("	")[:3] # 获取前两列的数据：日期和天气
                    date_weather[1] = date_weather[1].split("/")[0].strip() # 天气中包含白天和晚上，截取白天的天气
                    temperature = date_weather[2].split("/")
                    date_weather[2] = temperature[0].replace("℃",'').strip()
                    date_weather.append(temperature[1].replace("℃",'').strip())
                    text = ",".join(date_weather) + "\n"
                    fout.write(text)

    def read_tourist(self, file_in):
        """
        从excel表格读取游客数据
        :param file_in:
        :return: 返回游客数量字典，key是日期，value是客流量
        """
        workbook = xlrd.open_workbook(file_in)
        sheet = workbook.sheet_by_index(0)
        tourist_dict = {}  #客流信息字典，key是日期，value是客流量
        for row in range(4, sheet.nrows - 1):
            day = sheet.cell(row,0).value
            if sheet.cell(row,0).ctype == 3:
                value = xldate_as_tuple(sheet.cell(row,0).value,0)
                day = datetime.datetime(*value).strftime('%Y-%m-%d') #将日期转换为"2019-08-08"的格式

            # tourist_dict[day] = int(sheet.cell(row,1).value) + int(sheet.cell(row,2).value) #总的游客数(两列)
            if (sheet.cell(row,1).value != ""):
                tourist_dict[day] = int(sheet.cell(row,1).value) #总的游客数（一列)
            else:
                tourist_dict[day] = None

        return tourist_dict

    def read_weather2(self, file_in):
        """
        从excel表格读取天气综合数据,key是日期，value是[最低温度,最高温度，平均温度，湿度，风速，总降水量，平均总云量]
        min_temperature,max_temperature,mean_temperature,humidity,wind_speed,precipitation,cloudage
        :param file_in:
        :return:
        """
        workbook = xlrd.open_workbook(file_in)
        sheet = workbook.sheet_by_index(0)
        weather_dict = {}
        index = [2,3,4,5,6,10,11]
        for i in range(1,sheet.nrows - 1):
            row_list = sheet.row_values(i)
            date = row_list[1]
            # if sheet.cell(i, 0).ctype == 3: #excel文档的ctype 分为5种，对应分别是：empty: 0,string: 1,number:2,date: 3,boolean :4,error:5
            #     value = xldate_as_tuple(sheet.cell(i, 0).value, 0)
            #     date = datetime.datetime(*value).strftime('%Y-%m-%d')
            weather_dict[date] = [row_list[i] for i in range(len(row_list)) if i in index]
        # print(weather_dict["2016-10-01"]) # 空值''未处理
        return weather_dict

    def read_weather(self, file_in):
        """
        将天气数据转换为字典形式，key是日期，value是[天气,最高温度，最低温度]
        :param file_in:
        :return:
        """
        weather_dict = {}
        with codecs.open(file_in,'r', 'utf-8') as fin:
            for line in fin.readlines():
                date_weather = line.strip().split(",")
                date = date_weather[0].replace(r'年','-').replace(r'月','-').replace(r'日','') #将日期中的"年、月、日"替换为"-"
                try:
                    weather_dict[date] = [date_weather[1],date_weather[2],date_weather[3]]
                except IndexError:
                    print(weather_dict[date], date)
        return weather_dict

    def read_holiday(self, holiday_file):
        """
        将节假日数据转换为字典形式，key是日期，value是节假日信息
        :param holiday_file: 从api中获取的节假日json文件
        :return: 返回一个字典，包含所有节假日和调休日，key是日期，value是一个布尔值，true表示为节假日，false表示为调休(工作)
        """
        holiday_dict = {}
        with open(holiday_file, 'r', encoding='utf-8') as f:
            file_dict = json.load(f)
        year = file_dict['year']
        for day in file_dict['holiday']:
            holiday_dict[year+'-'+day] = file_dict['holiday'][day]['holiday']
        return holiday_dict


if __name__ == "__main__":
    file_in = "../data/weather_data/2018weather.txt"
    file_out = "../data/weather/2018weather.csv"
    excel = "../data/2018年灵山景区旅游人数统计汇总表.xlsx"
    # Reader().get_weather(file_in, file_out)
    Reader().read_weather2("../data/weather_data/2016-2019weather.xlsx")
    # result = Reader().read_tourist(excel)
    # print(result)
    # holiday_file = "../data/2015holiday.json"
    # result = Reader().read_holiday(holiday_file)
    # print(result)
    pass