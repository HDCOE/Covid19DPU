'''
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator


#calib_image_list = './build/quantize/images/calib_list.txt'
#calib_batch_size = 10



def crop_top(img, percent=0.15):
  offset = int(img.shape[0] * percent)
  return img[offset:]

def central_crop(img):
  size = min(img.shape[0], img.shape[1])
  offset_h = int((img.shape[0] - size) / 2)
  offset_w = int((img.shape[1] - size) / 2)
  return img[offset_h:offset_h + size, offset_w:offset_w + size]

def process_image_file(filepath, top_percent, size):
  img = cv2.imread(filepath)
  img = crop_top(img, percent=top_percent)
  img = central_crop(img)
  img = cv2.resize(img, (size, size))
  return img

def calib_input(iter):
  calib_image_list = 'test_split_250.txt'
  calib_batch_size = 8
  testfolder = 'Trainset/test'
  images = []
  line = open(calib_image_list).readlines()

  lentest = len(line)
  for i in range(0, calib_batch_size):
    ld = line[i].split()
    image = process_image_file(os.path.join(testfolder, ld[1]), 0.08, 224)
    image = image/255.0
    images.append(image)

  return{"images_in": images}
