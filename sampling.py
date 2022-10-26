import numpy as np
import random
from options import *
from torchvision import datasets, transforms


def get_img_num_per_cls(dataset,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (50000-num_meta)/10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000-num_meta)/100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def get_train_data(train_dataset,dset,args):
    img_num_list = get_img_num_per_cls(dset, args.imb_factor, args.num_meta * args.num_classes)
    random.shuffle(img_num_list)
    data_list_val = {}
    for j in range(args.num_classes):
        data_list_val[j] = [i for i, label in enumerate(train_dataset.targets) if label == j]
    idx_to_meta = []
    print('img_num_list:{},length:{}'.format( img_num_list,len(img_num_list)))
    im_data=[]
    for cls,img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        idx_to_meta.extend(img_id_list[:args.num_meta])
        img_id_list=np.delete(img_id_list,np.arange(args.num_meta))
        train_data=np.random.choice(img_id_list,img_num_list[cls],replace=False)
        im_data.extend(train_data)
        data_list_val[cls]=set(train_data)
    random.shuffle(idx_to_meta)
    random.shuffle(im_data)
    return im_data,idx_to_meta,data_list_val,img_num_list

def Divide_groups(dataset_train,train_list,dict_per_cls,num_users,args):
    class_num=np.arange(args.num_classes)
    imb_cls_num=list(np.random.randint(2,args.num_classes+1,(args.num_users)))
    num_per_user=int(len(train_list)/args.num_users)
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    num_cls = 0
    while(num_cls<args.num_users):
        user_set = set()
        while(len(user_set)<num_per_user):
            user_cls = np.random.choice(class_num, imb_cls_num[num_cls], replace=False)
            for j in range(len(user_cls)):
                user_set = user_set | dict_per_cls[user_cls[j]]
        dict_users[num_cls] = np.random.choice(np.array(list(user_set)), num_per_user, replace=False)
        random.shuffle(dict_users[num_cls])
        for m in range(len(user_cls)):
            dict_per_cls[user_cls[m]]-=user_set&set(dict_users[num_cls])
        num_cls+=1
    return dict_users