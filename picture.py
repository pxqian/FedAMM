import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from resnet import VNet
from matplotlib import style
from mpl_toolkits.axisartist.axislines import Subplot

def pic_data(train_dataset,dict_users,args):
    dict_users_final={i:[] for i in range(args.num_users)}
    user_id=np.arange(args.num_users)
    np.random.shuffle(user_id)

    for u_id in range(args.num_users):
        dict_users_final[u_id]=dict_users[user_id[u_id]]

    dict_label = {i: [] for i in range(args.num_users)}
    dict_list = {i: [] for i in range(args.num_users)}
    # # file_name='sample.txt'
    for i in range(args.num_users):
        for j in range(len(dict_users_final[i])):
            dict_label[i].append(train_dataset.targets[dict_users_final[i][j]])
            dict_label[i].sort()
        # cls_list=[]
        # for m in range(len(dict_label[0])):
    for i in range(args.num_users):
        for cls in range(args.num_classes):
            dict_list[i].append(list(dict_label[i]).count(cls))
    labels=[]
    for u in range(args.num_users):
        labels.append(str(u))
    # ,'5','6','7','8','9']
    bottom_num= {i: [0]*len(labels) for i in range(args.num_classes+1)}
    cls_num = {i: [] for i in range(args.num_classes )}
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')


    for cls in range(args.num_classes):
        for u in range(len(labels)):
            bottom_num[cls+1][u]=bottom_num[cls][u]+dict_list[u][cls]

    for cls in range(args.num_classes):
        for u in range(len(labels)):
            cls_num[cls].append(dict_list[u][cls])



    width = 0.6
    plt.cla()
    plt.figure(dpi=600.0)

    for cls in range(args.num_classes):

        plt.bar(labels, cls_num[cls], width,bottom=bottom_num[cls],label='class '+str(cls))
        # ax.bar(labels, cls_num[2], width, bottom=cls_num[1]+cls_num[0])


    plt.axis('off')


    plt.savefig('./runs/data_pic/{}_{}.png'.format(args.dataset,  current_time),bbox_inches = 'tight')

