from tool.utils import *
from global_variable import *
from collections import defaultdict
from sklearn.metrics import auc
from matplotlib import pyplot as plt
#from sklearn.metrics import average_precision_score as AP

txt_save_path="./evaluation"
#--------------------------------------------------------
# 先將demo.py存出來的'yolo_predict15.txt'做nms,然後存成'yolo_pred15_nms.txt'

def AP(y_true, y_conf, return_pr=False):
    global precisions, recalls
    data = [y for _,y in sorted(zip(y_conf, y_true), reverse=True)] #sort from highest conf to the lowest
    number_of_true = sum(data)
    number_of_false = len(data)-number_of_true
    recalls=[]
    precisions=[]
    tp=0
    fp=0
    N=0
    for datum in data:
        N+=1
        if datum:
            tp += 1
        else:
            fp += 1
        curr_precision = tp/N
        #print(curr_precision)
        curr_recall = tp/number_of_true
        if len(precisions)==0:
            precisions.append(curr_precision)
            recalls.append(curr_recall)
        else:
            precisions.append(curr_precision)
            recalls.append(curr_recall)
    #get AUC
    ap = auc(recalls, precisions)
    if return_pr:
        return ap,recalls,precisions
    else:
        return ap

def apply_nms(video=13):
    inpf=txt_save_path+'/yolo_pred13.txt'.format(video)
    outf=txt_save_path+'/yolo_pred13_nms.txt'.format(video)
    pred = []
    with open (inpf,'r') as file:
        for line in file:
            line = line.strip().split()
            line = line[0:10]
            pred.append(line)
    print(len(pred))    # 2124

    #turn to nms
    dict_ = defaultdict(list)
    for i in range(len(pred)):
        frame, prob, id, _, _, _, left, top, right, bottom = pred[i]
        
        box = [float(left), float(top), float(right), float(bottom), float(prob)]
        dict_[frame].append(box)

    with open(outf,'w') as f:
        for frame, boxes in dict_.items():
            # print(frame)
            x = nms(boxes,0.5)
            for box in x:
                left, up, right, bottom, prob = box           
                f.write(str(frame) + ' '+ str(prob)+ ' ' + 'person' + ' '+ '0' +' '+ '0' +' '+ '0'+ ' '+str(left) +' '+ str(up)+' '+str(right)+' '+str(bottom)+'0'+' '+'0'+' '+'0'+' ' +'0'+' '+'0'+' '+'0'+' '+'0'+' '+'0'+' '+'\n')

def calculate_iou_recall(videos=[15]):
    tps, fps, fns, all_ious, N = [],[],[],0,0
    y_trues, y_confs = [], []
    for video in videos:
        label = []
        y_true, y_conf = [], []
        with open (KITTI_LABEL_DIR+'/{:04d}.txt'.format(video),'r') as file:
            for line in file:
                line = line.strip().split()
                if line[2]=='Pedestrian' or line[2]=='Person_sitting' or line[2]=='Cyclist':
                    line = line[0:10]
                    label.append(line)
        #print(len(label))
        pred_box = []
        count = 0
        all_iou = 0
        #with open (txt_save_path+'/yolo_pred{}_nms.txt'.format(video),'r') as file: #NMS
        with open (txt_save_path+'/yolo_pred{}.txt'.format(video),'r') as file:
            for line in file:
                line = line.strip().split()
                line = line[0:10]
                pred_box.append(line)
        # print(len(pred_box))
        tp, fp, fn = 0, 0, 0
        for i in range(len(pred_box)):
            frame, prob, id, _, _, _, left, top, right, bottom = pred_box[i]    
            pbox = [float(left), float(top), float(right), float(bottom)]
            find = False
            best_iou = 0
            for j in range(len(label)):
                frame_, _, id_, _, _, _, left_, top_, right_, bottom_ = label[j]
                gbox = [float(left_), float(top_), float(right_), float(bottom_)]
                if int(frame) == int(frame_):
                    iou = bbox_iou(pbox, gbox)
                    best_iou = max(best_iou, iou)           
                    if iou > 0.5:
                        find = True
                        break
            all_iou += best_iou
            if find == False:
                fp+=1
                y_true.append(0)
                y_conf.append(float(prob))
            elif find == True:
                tp+=1
                y_true.append(1)
                y_conf.append(float(prob))
                
        for j in range(len(label)):
            frame_, _, id_, _, _, _, left_, top_, right_, bottom_ = label[j]
            gbox = [float(left_), float(top_), float(right_), float(bottom_)]
            not_found = True
            for i in range(len(pred_box)):
                frame, prob, id, _, _, _, left, top, right, bottom = pred_box[i]
                pbox = [float(left), float(top), float(right), float(bottom)]
                if frame == frame_:
                    iou = bbox_iou(pbox, gbox)
                    if iou>0.5:
                        not_found = False
                        break
            if not_found == True:
                fn += 1
        # print(all_iou)
        print("***** Video:{} *****".format(video))
        print('average iou:', all_iou/len(pred_box))
        print('tp:',tp)
        print('fp:',fp)
        print('fn:',fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        print('precision: ',precision)
        print('recall: ',recall)
        print('AP:', AP(y_true, y_conf))
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        all_ious += all_iou
        N += len(pred_box)
        y_trues += y_true
        y_confs += y_conf
        
    print("=================Total=================")
    tps,fps,fns = sum(tps),sum(fps),sum(fns)
    print('average iou:', all_ious/N)
    print('tp:',tps)
    print('fp:',fps)
    print('fn:',fns)
    precision = tps/(tps+fps)
    recall = tps/(tps+fns)
    print('precision: ',precision)
    print('recall: ',recall)
    try:
        ap, r, p = AP(y_trues, y_confs, return_pr=True)
        print('AP:', ap)
        plt.plot(r,p)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precison-Recall Curve")
        plt.savefig(txt_save_path+"/PR_curve.png")
        plt.show()
    except:
        print('AP:', AP(y_trues, y_confs))

#----------------------------
# 五筆資料分別的tp,fp,fn
# tp = 537+342+679+342+2491
# fp = 568+1744+146+207+206
# fn = 615+820+1430+495+3502
# precision = tp/(tp+fp)
# recall = tp/(tp+fn)
# print('precision: ',precision)
# print('recall: ',recall)


if __name__ == "__main__":
    pass
    calculate_iou_recall(videos=[13,15,16,17,19])
    #apply_nms(19)
    #calculate_iou_recall(videos=[19])
