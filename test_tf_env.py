# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:52:51 2020

@author: rodrigo
"""

import tensorflow as tf

print("Tensor Flow version: {}".format(tf.__version__))

if tf.config.list_physical_devices('GPU'): 
    print('Default Device: {}'.\
          format(tf.config.list_physical_devices('GPU')[0]))
else:
   print("Please install GPU version of TF")

# tf.test.gpu_device_name()
