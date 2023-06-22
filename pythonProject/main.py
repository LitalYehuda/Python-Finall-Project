# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import functions as fun
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    dataOld = pd.read_csv('data.csv')
    data2 = pd.read_csv('data-2.csv')
    data3 = pd.read_csv('data-3.csv')
    data1 = fun.readDataAndfix(dataOld)

     #שאילתות רגילות
    fun.Q1(data2)
    fun.Q2(data1)
    fun.Q3(data1)
    fun.Q4(data2)
    fun.Q5(data3)
    fun.Q6(data3)
    fun.Q7(data2)
    fun.Q8(data1)
    fun.Q9(data3)
    fun.Q10(data3)

    #שאילתות להשוואה
    fun.Q11(data2)
    fun.Q12(data1, data3)
    fun.Q13(data1)
    fun.Q14(data1)
    fun.Q15(data1)
    fun.Q16(data2)
    fun.Q17(data2)
    fun.Q18(data2)
    fun.Q19(data1)
    fun.Q20(data2)

    #דשבורד
    fun.Q21(data2)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
