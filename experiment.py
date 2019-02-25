import os

for layer in range(10):
    for num in range(10):
        os.system('python3 main.py --Dlayer={}'.format(layer+1))
        print(str(layer+1)+'層'+str(num+1)+'回目')