import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Perceptron
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

'''寻找最优阶数'''
def found_best_stage(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rmses = []
    degrees = np.arange(1, 10)
    min_rmse, min_deg,score = 1e10, 0 ,0

    for deg in degrees:
        # 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        x_train_poly = poly.fit_transform(x_train)

        # 多项式拟合
        poly_reg = LinearRegression()
        poly_reg.fit(x_train_poly, y_train)
        #print(poly_reg.coef_,poly_reg.intercept_) #系数及常数
        
        # 测试集比较
        x_test_poly = poly.fit_transform(x_test)
        y_test_pred = poly_reg.predict(x_test_poly)
        
        #mean_squared_error(y_true, y_pred) #均方误差回归损失,越小越好。
        poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        rmses.append(poly_rmse)
        # r2 范围[0，1]，R2越接近1拟合越好。
        r2score = r2_score(y_test, y_test_pred)
        
        # degree交叉验证
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg
            score = r2score
        print('degree = %s, RMSE = %.2f ,r2_score = %.2f' % (deg, poly_rmse,r2score))
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(degrees, rmses)
    ax.set_yscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('RMSE')
    ax.set_title('Best degree = %s, RMSE = %.2f, r2_score = %.2f' %(min_deg, min_rmse,score))  
    plt.show()

def key_data():
    #定义x、y散点坐标
    ploss=[
        [0, 10,25,40,60],
        [0, 10, 25, 40, 50],
        [0, 5, 10, 25, 50],
        [0, 5, 10, 15, 30],
        [0, 10, 20, 30, 40, 50, 60],
        [0, 15, 25, 50, 65],
        [0, 15, 25, 50, 65],
        [0, 15, 25, 30, 50, 60]
    ]
    # pl_bx_x = [0,5,10,25,40,50]
    # cr_x = [ 5, 10, 25, 50]
    # sb = [0, 5, 10, 15, 30]
    # dp = [0, 10, 20, 30, 40, 50, 60]
    # obs = [0, 15, 25, 30, 50, 60]
    util_value = [
        [0, 0.5, 2, 5, 10],
        [0, 1, 2, 5, 10],
        [0, 0.1, 0.5, 2, 5],
        [0, 0.1, 0.5, 2, 5],
        [0, 0.25, 0.5, 1, 2, 5,10],
        [0, 0.1, 2, 5, 10],
        [0, 0.5, 2, 5, 10],
        [0, 1, 3, 4, 5, 10]
    ]

    for i in range(8):
        x = np.array(ploss[i]).reshape(-1, 1)
        y = np.array(util_value[i]).reshape(-1, 1)
        fun0=0.0023*x*x+0.03*x-0.05
        fun1=0.00016*pow(x,3)-0.0074*pow(x,2)+0.16*x-0.2
        fun2=0.111*x-0.5 # x: 0-50, y: 0-5
        fun3=0.0037*pow(x,2)+0.063*x-0.16 # x:0-30 y: 0-5
        fun4=0.0001*pow(x,3) - 0.0048*pow(x,2) + 0.08*x - 0.05
        fun5=0.0026*pow(x,2) - 0.03*x
        fun6=0.002*pow(x,2) - 0.1 # // 0.1x-0.5
        fun7=0.0015*pow(x,2)+0.06*x+0.12
        y_pred = locals()['fun'+str(i)]
        r2=R2_0(y, y_pred=y_pred)
        print("=====process defect ", i, " R2=", r2)
    

def R2_0(y_test,y_pred):
    SStot=np.sum((y_test-np.mean(y_test))**2)
    SSres=np.sum((y_test-y_pred)**2)
    r2=1-SSres/SStot
    return r2


# key_data()

def draw_hough_space():
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    harvest = np.array([[1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 0, 0, 3, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 2, 0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 2, 0, 0]])


    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    # ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                        ha="center", va="center", color="w")

    # ax.set_title("Harvest of local farmers (in tons/year)")
    ax.set_xlabel('x', loc='right')
    ax.set_ylabel('y', loc='top')
    fig.tight_layout()
    plt.show()

# draw_hough_space()