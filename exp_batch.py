import os

for batch in [1, 4, 8, 16, 32, 64, 128, 256, 512]:
    for num in range(10):
        os.system('python3 main.py --batch_size={}'.format(batch))
        print(str(batch)+'層'+str(num+1)+'回目')