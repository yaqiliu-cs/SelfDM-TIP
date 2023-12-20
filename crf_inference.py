#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:20:52 2019

@author: root
"""


import convcrf

import torch
from torch.autograd import Variable

def do_crf_inference(image, unary):


    num_classes = unary.shape[1]
    shape = image.shape[2:4]
    # get basic hyperparameters
    config = convcrf.default_conf
    config['filter_size'] = 7

    img_var = Variable(torch.Tensor(image)).cuda()
    unary_var = Variable(torch.Tensor(unary)).cuda()
    ##
    # Create CRF module
    gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes)
    # Cuda computation is required.
    # A CPU implementation of our message passing is not provided.
    gausscrf.cuda()
    # Perform CRF inference
    prediction = gausscrf.forward(unary=unary_var, img=img_var, num_iter=1)

    return prediction