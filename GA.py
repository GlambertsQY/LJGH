# -*- coding: UTF-8 -*-
'''*******************************
@ 开发人员：Mr.Zs
@ 开发时间：2020/5/18 19:22
@ 开发环境：PyCharm
@ 项目名称：算法类总工程->遗传算法路径规划V1.0.py
******************************'''
import random   #生成随机整数
import numpy as np #生成随机小数
import math #用于计算除法 取整等运算
print(r'''
遗传算法是对参数集合的编码而非针对参数本身开始进化
遗传算法是从问题解的编码组开始，而非单个解开始搜索

step1:建立地图
step2:初始化种群（随机生成若干条从起点能够到达终点的路径(可行解)，每一条可行路径为一个个体）
step3:计算个体的适应度值
step4:选择适应度合适的个体进入下一代
step5:交叉
step6:变异
step7:更新种群，若没有出现最优个体，则转至step3
step8:输出最优的个体作为最优解
参考文献：https://blog.csdn.net/qq_40870689/article/details/86916910?ops_request_misc=&request_id=&biz_id=102&utm_term=%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95%E8%B7%AF%E5%BE%84%E8%A7%84%E5%88%92&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-86916910
''')

#这个对象专门滤除路径中的回头路
class Fiter:
    def __init__(self):
        self.b = 1  # 标志位

    def function(self, a):  # 定义一个函数
        for i in a:  # 遍历列表中的内容
            a = a[a.index(i) + 1:]  # 把当前内容索引的后面的内容剪切下来  因为前面的已经比对过了
            if i in a:  # 如果当前内容与后面有重复
                return i, 1  # 返回当前重复的内容 以及标志位1
            else:  # 没有重复就不用管  继续for循环
                pass
        return 0, 0  # 全部遍历完  没有重复的就返回0 这里返回两个0 是因为返回的数量要保持一致

    def fiter(self, a):
        while (self.b == 1):  # 标志位一直是 1  则说明有重复的内容
            (i, self.b) = self.function(a)  # 此时接受函数接收 返回值 i是重复的内容 b是标志位
            c = [j for j, x in enumerate(a) if x == i]  # 将重复内容的索引全部添加进c列表中
            a = a[0:c[0]] + a[c[-1]:]  # a列表切片在重组
        return a
fiter = Fiter()#实例化对象
#将地图上的点抽象话，可以用x表示横坐标 y表示纵坐标
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other): #函数重载  #判断两个坐标  的  值 是否一样
        if((self.x == other.x )and (self.y == other.y)):
            return  True
        else:
            return False
    def __ne__(self, other):
        pass

#step1
class Map():
    '''
    :param:地图类
    :param:使用时需要传入行列两个参数 再实例化
    '''

    def __init__(self,row,col):
        '''
        :param:row::行
        :param:col::列
        '''
        self.start_point = Point(0,0)
        self.end_point = Point(row-1,col-1)
        self.data = []
        self.row = row
        self.col = col
    def map_init(self):
        '''
        :param:创建栅格地图row行*col列 的矩阵
        '''
        self.data = [[0 for i in range(self.col)] for j in range(self.row)]
        # for i in range(self.row):
        #     for j in range(self.col):
        #         print(self.data[i][j],end=' ')
        #     print('')
    def map_Obstacle(self,num):
        '''
        :param:num:地图障碍物数量
        :return：返回包含障碍物的地图数据
        '''
        self.num = num
        for i in range(self.num):#生成小于等于num个障碍
            self.data[random.randint(0,self.row-1)][random.randint(0,self.col-1)] = 1
        if self.data[0][2] == 1:        #判断顶点位置是否是障碍  若是 修改成可通行
            self.data[0][0] = 0

        if self.data[self.row-1][0] == 1:
            self.data[self.row-1][0] = 0

        if self.data[0][self.col-1] == 1:
            self.data[0][self.col - 1] = 0

        if self.data[self.row-1][self.col - 1] == 1:
            self.data[self.row - 1][self.col - 1] = 0
        for i in range(self.row):       #显示出来
            for j in range(self.col):
                print(self.data[i][j],end=' ')
            print('')
        return self.data
#step2 初始化种群 也是最难的一步
class Population():
    def __init__(self,row,col,num,NP):
        self.row = row
        self.col = col
        self.num = num
        self.NP = NP
        self.p_start = 3  # 起始点序号  10进制编码
        self.p_end = self.row * self.col - 1  # 终点序号 10进制编码
        self.xs = (self.p_start) // (self.col)  # 行
        self.ys = (self.p_start) % (self.col)  # 列
        self.xe = (self.p_end) // (self.col)  # 终点所在的行
        self.ye = (self.p_end) % (self.col)
        self.map_start = Map(self.row, self.col)  # Map的实例化 主函数中可以不再初始化地图
        self.map_start.map_init()  # map初始化 生成0矩阵
        self.map = self.map_start.map_Obstacle(self.num)  # 生成带障碍的map

        self.can = []  # 这个列表存放整个地图中不是障碍物的点的坐标 按行搜索
        self.popu = [[0 for i in range(self.col)]
                     for j in range(self.NP)]  # 种群的列表，包含NP个间断可行解，即从起点到终点的路径
        # self.popu = []#存放可行解  可行路径
        self.end_popu = []

    def Population_Init(self):
        '''
        :return:无返回值，作用是找出NP条不连续可行解
        '''


        for i in range(self.NP):       #添加NP条路径
            j = self.p_start-1
            for xk in range(0,self.row):  #多少行
                self.can = []             #清空can列表   用来缓存当前行的可行点
                for yk in range(j,self.col):   #从起点0 开始的列开始搜寻到终点的列  也就是此行中的自由点
                    num = (yk) + (xk) * self.col
                    if self.map_start.data[xk][yk] == 0:
                        self.can.append(num)
                # print(self.can,'自由点\n')
                length = len(self.can)     #此行的长度  随机生成指针取出坐标
                # print(length)
                self.popu[i][j] = (self.can[random.randint(j,length-1)])
                j += 1#j从1 开始  是因为 0 的元素时起点编号
            self.popu[i][0] = self.p_start  # 每一行的第一个元素设置为起点序号
            self.popu[i][-1] = self.p_end

            temp = self.Generate_Continuous_Path(self.popu[i])#将返回的一条路经 添加进end中
            if temp != []:      #将返回的空列表路径滤除
                temp = fiter.fiter(temp)    #滤除重复节点路径
                self.end_popu.append(temp)
            # print(self.end_popu, end='\n')
        # print('测试1',self.popu,end='\n')
        return self.end_popu
    # @staticmethod
    def Generate_Continuous_Path(self,old_popu):#生成连续的路径个体
        '''
        :param old_popu: 未进行连续化的一条路径
        :return:        无返回                   # 已经连续化的一条路径
        '''
        self.new_popu = old_popu    #传进来的参数就是一行的数组  一条路径
        self.flag = 0
        self.lengh = len(self.new_popu)  #先将第一条路经的长度取出来
        i = 0
        # print("lengh =",self.lengh )
        while i!= self.lengh-1:       #i 不等于 当前行的长度减去1  从0计数  这里有问题 待修改
            x_now = (self.new_popu[i]) // (self.col)  # 行 解码  算出此条路经的第i个元素的直角坐标
            y_now = (self.new_popu[i]) % (self.col)  # 列
            x_next =  (self.new_popu[i+1]) // (self.col) #计算此条路经中下一个点的坐标
            y_next =  (self.new_popu[i+1]) % (self.col)
            #最大迭代次数
            max_iteration = 0

            #判断下一个点与当前点的坐标是否连续 等于1 为连续
            while max(abs(x_next - x_now), abs(y_next - y_now)) != 1:
                x_insert = math.ceil((x_next + x_now) // 2)      #行
                y_insert = math.ceil((y_next + y_now) // 2) #ceil向上取整数     #列
                # print("x_insert = ",x_insert,"\ty_insert = ",y_insert)
                flag1 = 0

                if self.map_start.data[x_insert][y_insert] == 0:  #插入的这个坐标为0 可走
                    num_insert = (y_insert) + (x_insert) * self.col #计算出插入坐标的编码
                    self.new_popu.insert(i+1,num_insert)
                    # print(self.new_popu)
                    # print(num_insert)
                else:#插入的栅格为障碍  判断插入的栅格上下左右是否为障碍，以及是否在路径中，若不是障碍且不在路径中，就插入
                    #判断下方
                    if (x_insert + 1 < self.row)and flag1 == 0:       #保证坐标是在地图上的
                        if ((self.map_start.data[x_insert+1][y_insert] == 0)#下方不是障碍物
                            and (((y_insert) + (x_insert+1) * self.col) not in self.new_popu)):#编号不在已知路径中
                            num_insert = (y_insert) + (x_insert+1) * self.col  #计算下方的编号
                            self.new_popu.insert(i + 1, num_insert) #插入编号
                            flag1 = 1       #设置标志位 避免下面重复插入

                            # print('下方插入',num_insert)
                    #判断右方
                    if (y_insert + 1 < self.col)and flag1 == 0:  # 保证坐标是在地图上的 并且前面没有插入
                        if ((self.map_start.data[x_insert][y_insert+1] == 0)#右方不是障碍物
                            and (((y_insert+1) + (x_insert) * self.col) not in self.new_popu)):#编号不在已知路径中
                            num_insert = (y_insert+1) + (x_insert) * self.col  #计算右方的编号
                            self.new_popu.insert(i + 1, num_insert) #插入编号
                            flag1 = 1  # 设置标志位 避免下面重复插入
                            # print('右方插入',num_insert)
                    #判断上方
                    if (x_insert - 1 > 0) and flag1 == 0:  # 保证坐标是在地图上的
                        if ((self.map_start.data[x_insert-1][y_insert] == 0)#右方不是障碍物
                            and (((y_insert) + (x_insert-1) * self.col) not in self.new_popu)):#编号不在已知路径中
                            num_insert = (y_insert) + (x_insert-1) * self.col  #计算右方的编号
                            self.new_popu.insert(i + 1, num_insert) #插入编号
                            flag1 = 1  # 设置标志位 避免下面重复插入
                            # print('上方插入',num_insert)
                    #判断左方
                    if (y_insert - 1 > 0)and flag1 == 0:  # 保证坐标是在地图上的
                        if ((self.map_start.data[x_insert][y_insert-1] == 0)#右方不是障碍物
                            and (((y_insert-1) + (x_insert) * self.col) not in self.new_popu)):#编号不在已知路径中
                            num_insert = (y_insert-1) + (x_insert) * self.col  #计算右方的编号
                            self.new_popu.insert(i + 1, num_insert) #插入编号
                            flag1 = 1  # 设置标志位 避免下面重复插入
                            # print('左方插入',num_insert)
                    if flag1 == 0:  #如果前面没有插入新点  说明这条路径不对 删除
                        self.new_popu = []
                        break
                x_next = num_insert//self.col
                y_next = num_insert%self.col
                # x_next = x_insert
                # y_next = y_insert
                max_iteration += 1#迭代次数+1
                if max_iteration > 20:
                    self.new_popu = []  #超出迭代次数 说明此条路经可能无法进行连续   删除路径
                    break
            if self.new_popu == []:
                break
            self.lengh = len(self.new_popu)
            i = i+1
        # print(self.new_popu,'连续')
        return  self.new_popu#返回的是一条路径
#step3 计算适应度函数
def calvalue(popu,col):
    '''
    :param popu: 传入种群信息
    :param col: 这个参数是地图的列数 ，因为要解码用到
    :return:    返回的是每一条路径长度组成的列表
    '''
    hang = len(popu)#计算行
    value = [] #存放计算出来的路径长度值
    for i in range(hang):#从第0行开始 到最后一行
        value.append(0)
        single_popu = popu[i] #把当前行的元素赋给 single_popu 之后操作
        single_lengh = len(single_popu)#将当前行的长度拿出来
        for j in range(single_lengh-1):#从第一个元素计算到倒数第一个元素
            x_now = (single_popu[j]) // (col)  # 行 解码  算出此条路经的第i个元素的直角坐标
            y_now = (single_popu[j]) % (col)  # 列
            x_next = (single_popu[j + 1]) // (col)  # 计算此条路经中下一个点的坐标
            y_next = (single_popu[j + 1]) % (col)
            if abs(x_now - x_next) + abs(y_now - y_next) == 1:#路径上下左右连续 不是对角的 则路径长度为1
                value[i] = value[i]+1
            elif max(abs(x_now - x_next),abs(y_now - y_next))>=2:#惩罚函数 若跳跃或者穿过障碍
                value[i] = value[i] + 100
            else:
                value[i] = value[i]+1.4  #对角长度为根号2  即1.4
    return value
#step4 选择
def selection(pop,value):
    '''
    :param pop:种群
    :param value:适应度值列表
    :return:返回新的种群
    '''

    ###原来的方法会丢失种群数量
    now_value=[]#做倒数后的适应度值列表
    P_value = []  #存放适应度值占总比概率的列表
    random_deci = []#存放随机的小数
    new_popu = []       #选择后的种群
    sum_value = 0   #存放适应度的总和  计算轮盘赌的概率
    lengh = len(pop)#首先计算种群有多少行，也就是有多少初始路径
    for i in range(lengh):
        new_popu.append([]) #原始种群有多少个个体  新的种群就有多少
    for i in value:     #由于适应度值是距离  将需要的就是距离最短的  因此求倒数来排序
        now_value.append(1/i)  #将适应度取倒数   倒数越小适应度越小，路径越长  就可以按概率抛弃
        sum_value += (1/i)
    for i in now_value:#将每一个适应度值取出来  做比例  得出每个适应度值的占比概率 存到列表中
        P_value.append(i/sum_value)
    P_value = np.cumsum(P_value)#累加和 并排序 从小到大
    for i in range(lengh):
        random_deci.append(random.random())#产生 i个随即小数 存到列表中
    random_deci = sorted(random_deci)#从小到大排序
    fitin = 0
    newin = 0
    while(newin<lengh): #遍历每一行
        if random_deci[newin] < P_value[fitin]:
            new_popu[newin] = pop[fitin]
            newin += 1
        else:
            fitin += 1
    return new_popu
    # ####原来的方法会丢失种群数量
    # retain_rate = 0.6#适应度最优的保留几率
    # random_rate = 0.3#适应度差的保留几率
    # dict_my = dict(zip(value, pop))     #将路径距离和路径打包字典
    #
    # new_popu = []
    # sort_dis = sorted(dict_my)   #将字典按照键排序  也就是按照距离短-长排序
    # retain_lengh = int(len(pop)*retain_rate)      #适应度强的按照存活概率 保留下
    # temp = sort_dis[:retain_lengh]      #缓存下保留的距离，待会将距离对应的字典值取出来
    # # print(temp,'优秀保留')
    # for i in sort_dis[retain_lengh:]:   #距离长的按照随机概率保留
    #     if random.random() < random_rate:
    #         temp.append(i)
    # #temp现在存放的就是进入下一代的种群信息，他是距离信息，下一步通过字典将种群提取出来
    # for i in temp:
    #     new_popu.append(dict_my[i]) #至此种群选取完
    # return new_popu
#step5 交叉   拟采用单点交叉
def cross(parents,pc):
    '''
    :param parents: 交叉的父类
    :param pc:   交叉概率
    :return:
    '''
    children = []  #首先创建一个子代 空列表 存放交叉后的种群
    single_popu_index_list = []#存放重复内容的指针
    lenparents = len(parents)  #先提取出父代的个数  因为要配对 奇数个的话会剩余一个
    parity = lenparents % 2 #计算出长度奇偶性  parity= 1 说明是奇数个  则需要把最后一条个体直接加上 不交叉
    for i in range(0,lenparents-1,2):       #每次取出两条个体 如果是奇数个则长度要减去 一  range函数不会取最后一个
        single_now_popu = parents[i]   #取出当前选中的两个父代中的第一个
        single_next_popu = parents[i+1]#取出当前选中的两个父代中的第二个
        children.append([]) #子代添加两行  稍后存取新的种群
        children.append([])
        index_content = list(set(single_now_popu).intersection(set(single_next_popu))) #第一条路经与第二条路经重复的内容
        num_rep = len(index_content)          #重复内容的个数
        if random.random() < pc and num_rep>=3:
            content = index_content[random.randint(0,num_rep-1)]   #随机选取一个重复的内容
            now_index = single_now_popu.index(content)  #重复内容在第一个父代中的索引
            next_index = single_next_popu.index(content)#重复内容在第二个父代中的索引
            children[i] = single_now_popu[0:now_index + 1] + single_next_popu[next_index + 1:]
            children[i+1] = single_next_popu[0:next_index + 1] + single_now_popu[now_index + 1:]
        else:
            children[i] = parents[i]
            children[i+1] = parents[i+1]
    if parity == 1:     #如果是个奇数  为了保证种群规模不变 需要加上最后一条
        children.append([]) #子代在添加一行
        children[-1] = parents[-1]
    return children
#step6 变异
def mutation(children,pm):
    '''
    :param children: 子代种群
    :param pm: 变异概率
    :return: 返回变异后的新种群
    '''

    row = len(children)   #子代有多少行   即多少条路经
    new_popu = []
    for i in range(row):#提取出来每一行

        single_popu = children[i]
        if random.random()<pm:#小于变异概率   就变异
            col = len(single_popu)#每一行的长度 即列数  也就是这条路径有多少节点
            first = random.randint(1,col-2) #随机选取两个指针
            second = random.randint(1,col-2)
            if first != second :    #判断两个指针是否相同  不相同的话把两个指针中间的部分删除 在进行连续化
                #判断一下指针大小  便于切片
                if(first<second):
                    single_popu = single_popu[0:first]+single_popu[second+1:]
                else :
                    single_popu = single_popu[0:second] + single_popu[first+1:]
            temp = population.Generate_Continuous_Path(single_popu)#连续化
            if temp!= []:
                new_popu.append(temp)
        else:       #不变异的话就将原来的个体直接赋值
            new_popu.append(single_popu)
    return new_popu
if __name__ == '__main__':
    print('原始地图')
    population = Population(10,10,20,100)   #实例化对象 并生成初始带随机障碍的地图

    popu = population.Population_Init()  #种群初始化  得到np条初始可行路径
    # print(popu,'连续路径')#打印出np条可行路径

    for i in range(200):    #迭代200代
        lastpopu = popu #上次的种群先保存起来
        value = calvalue(popu,population.col) #计算适应度值
        # print(value,'适应度值')#打印出np条路径的适应度值

        new = selection(popu,value)#选择 按适应度返回新的种群
        # print(new,'新种群')
        # value = calvalue(new,population.col) #计算适应度值
        # print(value,'新种群适应度值')#打印出np条路径的适应度值
        child = cross(new,0.8)  #交叉  产生子代
        # print(child,'交叉子代')
        # value = calvalue(child,population.col) #计算适应度值
        # print(value,'子代适应度值')#打印出np条路径的适应度值
        popu = mutation(child,0.8)#变异 产生子代 并更新原始种群

        if popu == []:  #如果本次的种群成空了  则把上次的种群信息拿出来，并迭代结束
            popu = lastpopu
            break
        # print('第',i,'次迭代后的种群为：',popu)
    if popu == []:  #迭代完成后先判断 若迭代完成了 但是种群路径还是空 说明可能是没有路径
        print('无路径')
    else:
        value = calvalue(popu,population.col) #计算适应度值
        minnum = value[0]
        for i in range(len(value)):#找出最小的适应度函数值
            if value[i] < minnum:#小于最小适应度  则替代
                minnum = value[i]
        popu = popu[value.index(minnum)]#取出最小的适应度值的个体

        for i in popu:
            x = (i) // (population.map_start.col)  # 行 解码  算出此条路经的第i个元素的直角坐标
            y = (i) % (population.map_start.col)  # 列
            population.map_start.data[x][y] = '*'   #将路径用*表示
        print('\n规划地图')
        for i in range(population.map_start.row):   #显示路径
            for j in range(population.map_start.col):
                print(population.map_start.data[i][j],end=' ')
            print('')
        print('最短路径值为：',minnum)
        print('最短路径为：',popu)
