# -*- coding: utf-8 -*-
import os
import shutil


if __name__ == '__main__':
    path_trainset = "mnist/imgs_train"
    path_trainset_source = "mnist/imgs_train_source"
    path_trainset_target = "mnist/imgs_train_target"
    if not os.path.exists(path_trainset_source):
        os.mkdir(path_trainset_source)
    if not os.path.exists(path_trainset_target):
        os.mkdir(path_trainset_target)

    filenames = os.listdir(path_trainset)
    filenames.remove('img.txt')
    for i in range(len(filenames) // 2):
        print(i, filenames[i])
        shutil.copyfile(os.path.join(path_trainset, filenames[i]), os.path.join(path_trainset_source, filenames[i]))
    for i in range(len(filenames) // 2, len(filenames)):
        print(i, filenames[i])
        shutil.copyfile(os.path.join(path_trainset, filenames[i]), os.path.join(path_trainset_target, filenames[i]))
        
   
