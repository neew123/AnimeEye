from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

import glob
import random
import os
import numpy as np
import csv
import torch
import json

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

year_list = ["0001", "1996", "1997", "1998", "1999", "2000",
             "2001", "2002", "2003", "2004", "2005", "2006",
             "2007", "2008", "2009", "2010", "2011", "2012",
             "2013", "2014", "2015", "2017", "2019", "2019",
             "2020", "unknow"]

class ImageDataset(Dataset):
    def __init__(self, root, mode="train",base_size = 128,transforms_=None):
        self.transform = transforms_
        self.norm = transforms.Compose([
            transforms.ToTensor()
        ])

        #
        # read image
        self.files = []
        start_year = 5
        end_year = len(year_list)
        image_folder = os.path.join(root, 'datasets')
        #print("image folder",image_folder)
        mode_folder = os.path.join(image_folder, mode)
        for i in range(start_year, end_year):
            year = year_list[i]
            year_folder = os.path.join(mode_folder, year)
            #print(year_folder)
            img_files = glob.glob(os.path.join(year_folder, '*.jpg'))
            for file in img_files:
                self.files.append(file)
        #print(len(self.files))

        # read label file
        self.labels = dict()
        csv_file_1= '/home/yuanmengli/gitRepository/AnimeEncoder/datasets/attr.csv'
        with open(csv_file_1, "r") as csvfile:
            reader = csv.reader(csvfile)
            index = 0
            for line in reader:
                if index == 0:
                    pass
                else:
                    if self.labels.get(line[1]) is None:
                        self.labels[line[1]] = np.array(line[2:len(line)]).astype(np.float).tolist()
                index += 1
        print(len(self.labels))

        self.landmark = dict()
        json_train = '/home/yuanmengli/gitRepository/AnimeEncoder/datasets/train.json'
        with open(json_train,encoding='utf-8') as f:
            str1 = f.read()
            result_dict = eval(str1)
            for key in result_dict:
                #print("key",key) filename.jpg
                #print(type(key)) str
                # print(result_dict[key][0]['landmark']) list
                self.landmark[key] = result_dict[key][0]['landmark']

    def __getitem__(self, index):
        # load image
        file_path = self.files[index % len(self.files)]
        filename = os.path.basename(file_path)
        label = self.labels[filename]

        # get img
        image_pil = Image.open(file_path)
        img = self.norm(self.transform(image_pil))
        # get position
        land_mark = self.landmark[filename]
        land_arr = np.array(land_mark)
        eye_pos = [land_arr[14,0],land_arr[14,1],land_arr[19,0],land_arr[19,1]]
        eye_pos = torch.tensor(eye_pos)
        # get label
        # label = torch.tensor(label)
        # reference label
        return img,eye_pos

    def __len__(self):
        return len(self.files)

    def getLen(self):
        return self.__len__()

class TestDataset(Dataset):
    def __init__(self, root, mode="test",base_size=128,transforms_=None):
        self.transform = transforms_
        self.norm = transforms.Compose([
            transforms.ToTensor()
        ])
        #
        # read image
        self.files = []
        start_year = 5
        end_year = len(year_list)
        image_folder = os.path.join(root, 'dataset')
        #print("image folder",image_folder)
        mode_folder = os.path.join(image_folder, mode)
        for i in range(start_year, end_year):
            year = year_list[i]
            year_folder = os.path.join(mode_folder, year)
            #print(year_folder)
            img_files = glob.glob(os.path.join(year_folder, '*.jpg'))
            for file in img_files:
                self.files.append(file)
        #print(len(self.files))

        # read label file
        self.labels = dict()
        csv_file_1= '/home/yuanmengli/gitRepository/AnimeEncoder/datasets/attr.csv'
        with open(csv_file_1, "r") as csvfile:
            reader = csv.reader(csvfile)
            index = 0
            for line in reader:
                if index == 0:
                    pass
                else:
                    if self.labels.get(line[1]) is None:
                        self.labels[line[1]] = np.array(line[2:len(line)]).astype(np.float).tolist()
                index += 1
        print(len(self.labels))

        self.landmark = dict()
        json_train = '/home/yuanmengli/gitRepository/AnimeEncoder/datasets/test.json'
        with open(json_train,encoding='utf-8') as f:
            str1 = f.read()
            result_dict = eval(str1)
            for key in result_dict:
                #print("key",key) filename.jpg
                #print(type(key)) str
                # print(result_dict[key][0]['landmark']) list
                self.landmark[key] = result_dict[key][0]['landmark']

    def __getitem__(self, index):
        # load image
        file_path = self.files[index % len(self.files)]
        filename = os.path.basename(file_path)
        label = self.labels[filename]

        # get img
        image_pil = Image.open(file_path)
        img = self.norm(self.transform(image_pil))
        # get position
        land_mark = self.landmark[filename]
        land_arr = np.array(land_mark)
        eye_pos = np.array(land_arr[14,0],land_arr[14,1],land_arr[19,0],land_arr[19,1])
        eye_pos = torch.from_numpy(eye_pos)
        # get label
        label = torch.tensor(label)

        # reference label
        return img,eye_pos

    def __len__(self):
        return len(self.files)

    def getLen(self):
        return self.__len__()
