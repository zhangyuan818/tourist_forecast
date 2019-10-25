# _*_ coding: utf-8 _*_

"""
 @version: 1.0
 @author: ZhuJingrui
 @date: 2019/8/7
 @language：Python 3.6
"""
from datetime import datetime
import json
import requests
import time

"""
判断某天是否是节假日
"""

class IfHoliday():
    def __init__(self):
        pass

    def holiday_check(self, date):
        """
        判断某天是否是周末还是工作日，工作日返回0，普通周末返回1，节假日返回2
        :param date:  日期，格式为"2019-08-09"
        :return:
        """
        year = date[:4]
        assert int(year) >= 2017, '此API不支持2017年以前的节假日查询'
        holiday_api_url = "http://timor.tech/api/holiday/info/"
        req = requests.get(holiday_api_url + date).text
        holiday_info = json.loads(req)
        # print(holiday_info)
        if holiday_info["holiday"] !=  None:
            return 2
        elif holiday_info["holiday"] == None:
            day = datetime.strptime(date.replace('-', ''), "%Y%m%d").date()
            if day.weekday() in [5, 6]:
                return 1
            else:
                return 0
        time.sleep(3)

    def get_year_holiday(self, year, file_out):
        """
        将某年所有的节假日信息存入file_out文件
        :param year: 需要获取节假日的年份，如"2017" 只支持2017-当前年份
        :param file_out: 写入的文件名
        :return:
        """
        url = "http://timor.tech/api/holiday/year/"
        assert int(year) >= 2017, '此API不支持2017年以前的节假日查询'
        year = str(year)
        req = requests.get(url + year).text
        holiday_info = json.loads(req)
        holiday_info['year'] = year
        with open(file_out, 'w', encoding='utf-8') as f:
            json.dump(holiday_info, f, ensure_ascii=False)

    def holiday_check2(self, date):
        """
        判断某天是否是周末还是工作日，工作日返回0，普通周末返回1，节假日返回2
        与第一个函数不同的地方在于数据来源为本地数据
        :param date:  日期，格式为"2019-08-09"
        :return:
        """
        '''
        holiday_api_url = "http://timor.tech/api/holiday/info/"
        req = requests.get(holiday_api_url + date).text
        holiday_info = json.loads(req)
        print(holiday_info)
        if holiday_info["holiday"] !=  None:
            return 2
        elif holiday_info["holiday"] == None:
            day = datetime.strptime(date.replace('-', ''), "%Y%m%d").date()
            if day.weekday() in [5, 6]:
                return 1
            else:
                return 0
        time.sleep(3)
        '''
        pass


if __name__ == "__main__":
    # holiday = IfHoliday().holiday_check("2017-10-04")
    # today = datetime.now().strftime('%Y-%m-%d')
    # print(today)
    # print(holiday)
    IfHoliday().get_year_holiday("2018", '../data/2018holiday.json')
    pass