#read txt method one
import re
import numpy as np
import matplotlib.pyplot as plt

def load_file(path):
    #数据预处理
    #path = "data.txt"
    with open(path,'r') as f:
        ##定义一个分割正则
        seq = re.compile(" ")
        ans = []
        for l in f:
            ##逐行读取，l1[0]、l[1]分别表示topic数目，和coherencesore分数
            l1 = seq.split(l.strip())
            ans.append(l1)
    ##除去多余的[]
    ans = ans[:len(ans)-1]
    ##把字符串转化为float
    for i in range(len(ans)):
        for j in range(len(ans[0])):
            ans[i][j] = float(ans[i][j])
    print(type(ans[0][0]))
    return ans

def polynomial_fitting(ans,n):
    #ans表示n*2维的矩阵，第一列为x，第二列为y，n表示你想拟合的次数
    #对数据进行多项式拟合
    ###对数据进行分装
    print("哈哈！！！小姐姐，你用的是",n,"次的多项式拟合")
    print("\n")
    x = [ ans[i][0] for i in range(len(ans))]
    y = [ ans[j][1] for j in range(len(ans))]

    a = np.polyfit(x,y,n)                                                           #用n次多项式拟合x，y数组
    b = np.poly1d(a)                                                                #拟合完之后用这个函数来生成多项式对象
    print("打印拟合函数",b)
    print("\n")
    df_1 = np.poly1d.deriv(b)                                                       #拟合后这个函数的多项式的导数
    print("打印导函数",df_1)
    print("\n")
    c = b(x)                                                                        #生成多项式对象之后，就是获取x在这个多项式处的值
    d = df_1(x)                                                                     #多项式对象，获取x在这个多项式的导数值
    print("导数在x上的值",d)                                                          #导数在x上的值
    print("\n")
    plt.scatter(x,y,marker='o',label='original datas')                              #对原始数据画散点图
    plt.plot(x,c,ls='--',c='red',label='fitting with n-degree polynomial')     #对拟合之后的数据，也就是x，c数组画图
    plt.plot(x,d,ls='--',c = 'green',label = 'derivative function - x')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    path = "data.txt"
    ans = load_file(path)
    n = input("欢迎使用myky出品，请输入一个整数,来决定你用多少次的多项式拟合")
    print("\n")
    #n = 2
    polynomial_fitting(ans,int(n))



