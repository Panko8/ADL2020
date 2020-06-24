import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import OrderedDict
import os

def read_and_parse_history(file):
    history=OrderedDict()
    with open(file, 'r') as f:
        for line in f:
            if line == "\n":
                continue
            line=line.split()
            epoch = int(line[1])
            frac = float(line[3])
            loss = float(line[-1])
            history[epoch+frac] = loss
    return history
            
def plot_one(history, fig, axes, x_limit, window=1):
    assert type(window)==int, "bad window type, expect int"
    
    def moving_average(x, n) :
        rest = np.cumsum(x, dtype=float)
        rest[n:] = rest[n:] - rest[:-n]
        return rest[n-1:]/n

    x = list(history.keys())
    y = list(history.values())
    cut_off_idx=0
    for key in x:
        cut_off_idx += 1
        if key>x_limit:
            break
    
    x = moving_average(x[:cut_off_idx], window)
    y = moving_average(y[:cut_off_idx], window)
    axes.plot(x,y)
   

def main():
    global fig, axes
    SMOOTHING_WINDOW = 2 #2 for training, 5 for validation
    ENTRY = "training" #training/validation
    fig, axes = plt.subplots()
    fnames=[]
    histories=[]
    if len(sys.argv)==1:
        for f in os.listdir():
            if "history_" in f:
                sys.argv.append(f+"/%s.log"%ENTRY)
    for file in sys.argv[1:]:
        fname = file.split('\\')[-2] if '\\' in file else file.split('/')[-2]
        fname = fname[8:]
        fnames.append(fname)
        history = read_and_parse_history(file)
        histories.append(history)
    #x_limit = min( max(history.keys()) for history in histories )
    x_limit = 33
    #x_limit = float("inf")
    for history in histories:
        plot_one(history, fig, axes, x_limit, window=SMOOTHING_WINDOW)

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Learining curves for {}, smooth={}".format(ENTRY, SMOOTHING_WINDOW))
    leg = fig.legend(labels=fnames, bbox_to_anchor=[0.9,0.88], fontsize=8)
    plt.ylim(0, 7)
    plt.xlim(0, 35)
    #fig.save('./curve.png')
    fig.show()

if __name__ == "__main__":
    main()
