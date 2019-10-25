# _*_ coding: utf-8 _*_

"""
 @version: 1.0
 @author: ZhuJingrui
 @date: 2019/8/8
 @language：Python 3.6
"""
from tasks.create_data_task import DataCreator
from tasks.forcast_task import Forecast


def do_job():
    # 第一步：处理数据，目前对于缺失的数据当作0处理，合理的方式应该用拉格朗日插值法进行处理，后期可作改进优化
    # weather_file = "./data/2015weather.csv"  #天气文件
    # tourist_file = "./data/2015年灵山景区旅游人数统计汇总表.xlsx" #历史客流量数据文件
    processed_file = "./data/2015data.csv"   # 处理后的文件
    # DataCreator().create_data(weather_file, tourist_file, processed_file)
    # holiday_out = "../data/2015holiday.json"
    # DataCreator().create_data2(weather_file, tourist_file, 2015, holiday_out, file_out)

    # 第二步：随机森林预测
    date = "2018-10-08"  # 预测日期
    weather = "晴"       # 预测日期当天天气
    result = Forecast(processed_file).run(date, weather) # 预测结果
    print(result)


if __name__ == "__main__":
    do_job()
    pass