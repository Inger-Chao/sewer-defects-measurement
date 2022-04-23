#coding:utf-8
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from scipy.interpolate import make_interp_spline
from torch import sgn
plt.rcParams['font.sans-serif']=['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

'''效用函数
pl: y=0.0023*x*x+0.03*x-0.05
bx: y=0.00016*x^3-0.0074*x^2+0.16*x-0.2
cr: y=0.111*x-0.5, x: 0-50, y: 0-5
sb & pr: y=-0.004475*x^2+0.37*x-2.029 x:0-30 y: 0-5
dp: y=0.0001*x^3 - 0.0048*x^2 + 0.08*x - 0.05
obs: y= 0.0026*x^2 - 0.03*x
rt & jg: y= 0.002*x^2 - 0.1 // 0.1x-0.5
bw: y= 0.0015*x^2+0.06*x+0.12
'''
def fit_utility():
    #定义x、y散点坐标
    # pl_bx_x = [0,5,10,25,40,50]
    # cr_x = [ 5, 10, 25, 50]
    sb = [0, 5, 10, 15, 30]
    # dp = [0, 10, 20, 30, 40, 50, 60]
    # obs = [0, 15, 25, 30, 50, 60]

    x = np.array(sb)
    print('x is :\n',x)
    # pl = [0,0.1,0.5,2,5,10]
    # bx_y = [0,0.5,1,2,5,10]
    cr_y = [0, 0.1, 0.5, 2, 5]
    # dp_y = [0, 0.25, 0.5, 1, 2, 5,10]
    # obs_y = [0, 0.1, 2, 3, 5, 10]
    # rt_y=[0, 0.5, 2, 3, 5, 10]
    # bw_y = [0, 1, 3, 4, 5, 10]
    y = np.array(cr_y)
    print('y is :\n',y)
    # # 用3次多项式拟合
    f1 = np.polyfit(x, y, 2)
    print('f1 is :\n',f1)
    
    p1 = np.poly1d(f1)
    print('p1 is :\n',p1)
    
    #也可使用yvals=np.polyval(f1, x)
    # yvals = p1(x)  #拟合y值
    # print('yvals is :\n',yvals)


def draw_utility():
    x = np.arange(0, 60, 0.1)
    thirty_x = np.arange(0, 30, 0.1)
    fifty_x = np.arange(0, 50, 0.1)
    pl=0.0023*x*x+0.03*x-0.05
    bx=0.00016*pow(x,3)-0.0074*pow(x,2)+0.16*x-0.2
    cr=0.111*fifty_x-0.5 # x: 0-50, y: 0-5
    sb_pr=0.0037*pow(thirty_x,2)+0.063*thirty_x-0.16 # x:0-30 y: 0-5
    dp=0.0001*pow(x,3) - 0.0048*pow(x,2) + 0.08*x - 0.05
    obs= 0.0026*pow(x,2) - 0.03*x
    rt_jg= 0.002*pow(x,2) - 0.1 # // 0.1x-0.5
    bw= 0.0015*pow(x,2)+0.06*x+0.12

    #绘图
    plot1 = plt.plot(x, pl, 'lightcoral',label='pl')
    plot2 = plt.plot(x, bx, 'sienna',label='bx')
    plot3 = plt.plot(fifty_x, cr, 'red',label='fsh')
    plot4 = plt.plot(thirty_x, sb_pr, 'darkred', label='zgaj\nywchr')
    plot5 = plt.plot(x, dp, 'blue', label=('chj'))
    plot6 = plt.plot(x, obs, 'green', label=('zhaw'))
    plot7 = plt.plot(x, rt_jg, 'indigo', label=('shg\njg'))
    plot8 = plt.plot(x, bw, 'gold', label=('cqdb'))
    plt.xlim(0, 60)
    plt.ylim(0, 10)
    plt.xlabel('Ploss(%)')
    plt.ylabel(u'缺陷效用')
    plt.legend(loc=0) #指定legend的位置右下角
    plt.title(u'效用函数曲线')
    plt.show()

def svg_pl(x):
    if x < 10:
        return 0.5
    elif x < 25:
        return 2
    elif x < 60:
        return 5
    else:
        return 10

def svg_bx(x):
    if x < 10:
        return 1
    elif x < 25:
        return 2
    elif x < 50:
        return 5
    else:
        return 10

def svg_fsh(x):
    if x < 10:
        return 0.5
    elif x < 50:
        return 2
    else:
        return 5

def svg_zhgaj_ywchr(x):
    if x < 10:
        return 0.5
    elif x < 30:
        return 2
    else:
        return 5

def svg_chj(x):
    if x < 20:
        return 0.5
    elif x < 40:
        return 2
    elif x < 50:
        return 5
    else:
        return 10

def svg_zhaw(x):
    if x < 15:
        return 0.1
    elif x < 25:
        return 2
    elif x < 50:
        return 5
    else:
        return 10

def svg_shg_jg(x):
    if x < 15:
        return 0.5
    elif x < 25:
        return 2
    elif x < 50:
        return 5
    else:
        return 10

def svg_cqdb(x):
    if x < 15:
        return 1
    elif x < 25:
        return 3
    elif x < 50:
        return 5
    else:
        return 10

def sng_chj_uti(x):
    return min(0.0001*pow(x,3) - 0.0048*pow(x,2) + 0.08*x - 0.05, 10)

def sng_pl_uti(x):
    return min(0.0023*x*x+0.03*x-0.05, 10)

def sng_bx_uti(x):
    return min(0.00016*pow(x,3)-0.0074*pow(x,2)+0.16*x-0.2, 10)

def sng_fsh_uti(x):
    return min(0.111*x-0.5, 5)

def sng_zhgaj_ywchr_uti(x):
    return min(0.0037*pow(x,2)+0.063*x-0.16, 5)

def sng_zhaw_uti(x):
    return min(0.0026*pow(x,2) - 0.03*x, 10)

def sng_shg_jg_uti(x):
    return min(0.002*pow(x,2) - 0.1, 10)

def sng_cqdb_uti(x):
    return min(0.0015*pow(x,2)+0.06*x+0.12, 10)

# 分段函数和效用函数的区别：以沉积物缺陷为例
def draw_chj_diff():
    fig = plt.figure(figsize=(4, 4))
    ax = axisartist.Subplot(fig, 111)
    #将绘图区对象添加到画布中
    fig.add_axes(ax)
    x_cor = "管道损失度$P_{loss}(\%)$"
    y_cor = "缺陷效用"
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(nth_coord=0, value=0)#ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"].set_axisline_style("->", size = 1.0)#给x坐标轴加上箭头
    ax.axis["x"].label.set_text(x_cor)

    #添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1,0)
    ax.axis["y"].major_ticklabels.set_axis_direction("top")

    ax.axis["y"].set_axisline_style("->", size = 1.0)
    #设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("bottom")
    ax.axis["y"].set_axis_direction("left")
    ax.axis["y"].label.set_text(y_cor)

    ax.set_xlim(0, 72)
    ax.set_ylim(0, 10.5)

    x = np.linspace(0, 80, 1000)
    dp_y_org = np.array([])
    chj_y_uti = np.array([])
    for v in x:
        dp_y_org = np.append(dp_y_org, np.linspace(svg_zhaw(v), svg_zhaw(v), 1))
    for v in x:
        chj_y_uti = np.append(chj_y_uti, np.linspace(sng_zhaw_uti(v), sng_zhaw_uti(v), 1))
    chj_org_line = ax.plot(x, dp_y_org, 'black', linestyle='dashed', label="对照表缺陷分值")
    chj_uti_curve = ax.plot(x, chj_y_uti, 'black', label="效用函数")
    plt.legend(loc=0) #指定legend的位置右下角
    plt.show()


# fit_utility()
# draw_utility()
# draw_chj_diff()

for i in range(16):
    print(sng_zhaw_uti(i))