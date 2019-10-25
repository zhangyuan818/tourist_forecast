import numpy as np
import matplotlib.pyplot as plt
#三个参数分别是起始点，终止点和步长
x=np.arange(0.0,5,0.5)
y=np.exp(-x**2)
#误差条设置
error=0.1+0.5*x
#绘图是正常生成中文
plt.rcParams['font.sans-serif'] = ['SimHei']
#生成图纸，绘制两幅子图，分成两行，共享x轴坐标
fig,(ax0,ax1)=plt.subplots(nrows=2,sharex=True)
#绘制第一幅误差条形图，误差条分布方向为纵向，上下误差条对称分布
ax0.errorbar(x,y,yerr=error,fmt='-o')
ax0.set_title('对称的误差条形图')
#以y（x）轴方向为例，设置向下(左）误差条的长度
lower_error=0.3*error
#以y（x）轴方向为例，设置向上(右）误差条的长度
upper_error=error
#得到一个误差列表，为绘制非对称的误差条形图做准备
different_error=[lower_error,upper_error]
#绘制第二幅误差条形图
ax1.errorbar(x,y,xerr=different_error,fmt='o')
ax1.set_title('非对称的误差条形图')
#转变成对数刻度
ax1.set_yscale('log')plt.show()