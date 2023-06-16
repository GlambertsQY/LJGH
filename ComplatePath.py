import cmath

import numpy as np
#f = open(r"D:\data.txt")
fileHandler = open("data1.txt", "r")

listx = []
listy = []
listz = []
res = 0.0
ans = 0.0
goal = 0.0
def length():
    global res;
    # line = fileHandler.readline()
    # line = fileHandler.readline()
    # line = fileHandler.readline()
    while True:
        # Get next line from file
        # line = fileHandler.readline()
        # line = fileHandler.readline()
        line = fileHandler.readline()

        # If line is empty then end of file reached
        if not line:
            print(listx)
            print(listy)
            break;
        this_lines = line.split()  # 根据空格对字符串进行切割，由于切割后的数据类型有所改变(str-array)建议新建变量进行存储
        for index, this_line in enumerate(this_lines) :  # 遍历数组并输出
            if index%3==0:
                listx.append(float(this_line))
            if index%3==1:
                listy.append(float(this_line))
            # if index % 5 == 1:
            #         listx.append(float(this_line))
            # if index % 5 == 3:
            #         listy.append(float(this_line))
            print(this_line)
        # line = fileHandler.readline()
        # line = fileHandler.readline()
        # line = fileHandler.readline()

        # Close Close
    fileHandler.close()
    for i in range(len(listx) - 1):
        dx = (listx[i] - listx[i + 1]) ** 2
        dy = (listy[i] - listy[i + 1]) ** 2
        res += cmath.sqrt(dx + dy)
        print(res)


def rv():
    global ans;
    # while True:
    #     # Get next line from file
    #     line = fileHandler.readline()
    #     # If line is empty then end of file reached
    #     if not line:
    #         print(listx)
    #         print(listy)
    #         break;
    #     this_lines = line.split()  # 根据空格对字符串进行切割，由于切割后的数据类型有所改变(str-array)建议新建变量进行存储
    #     for index, this_line in enumerate(this_lines) :  # 遍历数组并输出
    #         if index%3==2:
    #             listz.append(float(this_line))
    #     # Close Close

    while True:
        # Get next line from file
        line = fileHandler.readline()
        line = fileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            print(listx)
            print(listy)
            break;
        this_lines = line.split()  # 根据空格对字符串进行切割，由于切割后的数据类型有所改变(str-array)建议新建变量进行存储
        for index, this_line in enumerate(this_lines):  # 遍历数组并输出
            if index % 5 == 4:
                listz.append(float(this_line))
        # Close Close

    fileHandler.close()
    for i in range(len(listz) - 1):
        dz = listz[i]-listz[i+1]
        ans += abs(dz)

    print(ans/len(listz))

def gd():
    global goal
    sum=0.0;
    # line = fileHandler.readline()
    # line = fileHandler.readline()
    # line = fileHandler.readline()
    while True:
        # Get next line from file
        line = fileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            print(listx)
            print(listy)
            break;
        this_lines = line.split()  # 根据空格对字符串进行切割，由于切割后的数据类型有所改变(str-array)建议新建变量进行存储
        for index, this_line in enumerate(this_lines):  # 遍历数组并输出
            if index % 3 == 2:
                listz.append(float(this_line))
            if index%3==0:
                listx.append(float(this_line))
            if index%3==1:
                listy.append(float(this_line))
            # if index % 5 == 1:
            #         listx.append(float(this_line))
            # if index % 5 == 3:
            #         listy.append(float(this_line))
            # if index % 5 == 4:
            #         listz.append(float(this_line))
        # line = fileHandler.readline()
        # line = fileHandler.readline()
        # line = fileHandler.readline()
        # Close Close
    fileHandler.close()
    for i in range(len(listz) - 1):
        if(abs(listz[i]-listz[i+1])>5):
            print("(",listx[i],",",listy[i],",",listz[i],")","(",listx[i+1],",",listy[i+1],",",listz[i+1],")")
            sum +=abs(listz[i]-listz[i+1])
            goal=goal+1

    print(goal,sum/goal)


if __name__ == "__main__":
      # length()
       gd()



