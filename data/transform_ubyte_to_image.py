# -*- coding: utf-8 -*-
import numpy as np
import struct
from PIL import Image
import matplotlib.pyplot as plt
import os
import imageio
# 不管是用PIL还是scipy还是imageio，都没有办法保存下来一个完完全全的二值图像，现在折中考虑
# 先保存成图片，然后在dataloader的时候给他按阈值127二值化一下吧。


class DataUtils(object):
    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath

        self._tag = '>'  # 大端格式
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte
        self._imgNums = 0
        self._LabelNums = 0

    def getImage(self):
        """
        将MNIST的二进制文件转换成像素特征数据
        """
        binfile = open(self._filename, 'rb') #以二进制方式打开文件
        buf = binfile.read()
        binfile.close()
        index = 0
        numMagic, self._imgNums, numRows, numCols = struct.unpack_from(self._fourBytes2, buf, index)
        index += struct.calcsize(self._fourBytes)
        images = []
        print('image nums: %d' % self ._imgNums)
        for i in range(self._imgNums):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j] >= 1:
                    imgVal[j] = 255
            images.append(imgVal)
        return np.array(images), self._imgNums

    def getLabel(self):
        """
        将MNIST中label二进制文件转换成对应的label数字特征
        """
        binFile = open(self._filename, 'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, self._LabelNums = struct.unpack_from(self._twoBytes2, buf, index)
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(self._LabelNums):
            im = struct.unpack_from(self._labelByte2, buf, index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def outImg(self, arrX, arrY, imgNums):
        """
        根据生成的特征和数字标号，输出png的图像
        """
        output_txt = self._outpath + '/img.txt'
        output_file = open(output_txt, 'a+')

        m, n = np.shape(arrX)
        # 每张图是28*28=784Byte
        for i in range(imgNums):
            img = np.array(arrX[i])
            img = img.reshape(28, 28).astype(np.uint8)
            outfile = str(i) + "_" + str(arrY[i]) + ".jpg"
            print('saving file: %s' % outfile)

            txt_line = self._outpath + '/' + outfile + " " + str(arrY[i]) + '\n'
            output_file.write(txt_line)

            # img = Image.fromarray(img, 'L')
            # img.save(self._outpath + '/' + outfile)
            imageio.imwrite(self._outpath + '/' + outfile, img) # PIL的保存有问题，很诡异，最后只能用imageio，即使它也没有完全
            print('saving file: %s; done' % outfile)

            # plt.figure()
            # plt.imshow(img, cmap='binary')  # 将图像黑白显示
            # plt.savefig(self._outpath + "/" + outfile)
        output_file.close()


if __name__ == '__main__':
    trainfile_X = 'mnist/train-images.idx3-ubyte'
    trainfile_y = 'mnist/train-labels.idx1-ubyte'
    testfile_X = 'mnist/t10k-images.idx3-ubyte'
    testfile_y = 'mnist/t10k-labels.idx1-ubyte'

    # 加载mnist数据集
    train_X, train_img_nums = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X, test_img_nums = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    # 以下内容是将图像保存到本地文件中
    path_trainset = "mnist/imgs_train"
    path_testset = "mnist/imgs_test"
    if not os.path.exists(path_trainset):
        os.mkdir(path_trainset)
    if not os.path.exists(path_testset):
        os.mkdir(path_testset)
    DataUtils(outpath=path_trainset).outImg(train_X, train_y, train_img_nums)
    DataUtils(outpath=path_testset).outImg(test_X, test_y, test_img_nums)
   
    # 原文：https://blog.csdn.net/m_buddy/article/details/80964194 
