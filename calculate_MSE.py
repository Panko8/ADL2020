import numpy as np
import pandas as pd
from collections import defaultdict
import sys

if len(sys.argv)!=2 or sys.argv[1] in ['-h', '--help']:
    raise TypeError("python calculate_MSE.py {prediction_file}")
    

EPOCH = 100
history = []
with open (sys.argv[1],'r') as file:
    for line in file:
        line = line.strip().split()
        history.append(line)


for epo in range(EPOCH):
    datas = defaultdict(list)
    for data in history[1:]:
        if int(float(data[1])) == epo:
            datas['epoch'].append(data[1])
            datas['file'].append(data[2])
            datas['frame'].append(data[3])
            datas['distance'].append(data[20])
            datas['ground truth'].append(data[22])

    if datas=={}:
        break #terminate

    ##MSE_0 = 0
    MSE_1 = 0
    MSE_2 = 0
    MSE_3 = 0
    ##count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for i, gt in enumerate(datas['ground truth']):
        ##if float(gt) >= 8:
        ##    MSE_1 += ((float(datas['distance'][i]) - float(datas['ground truth'][i]))**2)
        ##    count_0 += 1
        ##elif float(gt) >= 5:
        if float(gt) >= 3:
            MSE_1 += ((float(datas['distance'][i]) - float(datas['ground truth'][i]))**2)
            count_1 += 1
        ##elif float(gt) >= 5: ##1.5~5
        elif float(gt) >= 1.5: #1.5~3
            MSE_2 += ((float(datas['distance'][i]) - float(datas['ground truth'][i]))**2)
            count_2 += 1
        else: # <1.5
            MSE_3 += ((float(datas['distance'][i]) - float(datas['ground truth'][i]))**2)
            count_3 += 1
    print('epoch',epo)
    ##print('>8:', MSE_0 / count_0)
    ##print('5-8:', MSE_1 / count_1)
    print('>3:', MSE_1 / count_1)
    ##print('1.5-5:',MSE_2 / count_2)
    print('1.5-3:',MSE_2 / count_2)
    print('<1.5:',MSE_3 / count_3)



    
# from collections import defaultdict
# table = defaultdict(list)
# for index, (pred, label) in enumerate(zip(datas['distance'], datas['ground truth'])):
#     pred, label = float(pred), float(label)
#     if label >= 3:
#         table['3'].append([pred, label])
#     elif label >= 1.5:
#         table['1.5'].append([pred, label])
#     else:
#         table['0'].append([pred, label])

# import matplotlib.pyplot as plt
# maxes = [5, 10, 30]
# for index, key in enumerate(['0', '1.5', '3']):
#     table[key] = np.stack(table[key])
#     diff = ((table[key][:, 0] - table[key][:, 1])**2).mean()
#     print('Key : ', key)
#     print('MSELoss : ', diff)
#     print('Pred Distance Mean : ', table[key][:, 0].mean())
#     print('Pred Distance Std : ', table[key][:, 0].std())
#     print('Ground Truth Mean : ', table[key][:, 1].mean())
#     print('Ground Truth Std : ', table[key][:, 1].std())
    
#     plt.hist(table[key][:, 0], alpha=0.5, range=(0,maxes[index]), label='pred')
#     plt.hist(table[key][:, 1], alpha=0.5, range=(0,maxes[index]), label='gt')
#     plt.legend()
#     plt.show()
    
