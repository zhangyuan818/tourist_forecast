import numpy as np
import matplotlib.pyplot as plt
#创建自变量数组
x= np.linspace(0,2*np.pi,500)
#创建函数值数组
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x*x)
#创建图形
plt.figure(1)
'''
意思是在一个2行2列共4个子图的图中，定位第1个图来进行操作（画图）。
最后面那个1表示第1个子图。那个数字的变化来定位不同的子图
'''
#第一行第一列图形
ax1 = plt.subplot(2,2,1)
#第一行第二列图形
ax2 = plt.subplot(2,2,2)
#第二行
ax3 = plt.subplot(2,1,2)
#选择ax1
plt.sca(ax1)
#绘制红色曲线
plt.plot(x,y1,color='red')
#限制y坐标轴范围
plt.ylim(-1.2,1.2)
#选择ax2
plt.sca(ax2)
#绘制蓝色曲线
plt.plot(x,y2,'b--')
plt.ylim(-1.2,1.2)
#选择ax3
plt.sca(ax3)
plt.plot(x,y3,'g--')
plt.ylim(-1.2,1.2)
plt.show()