#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:00:46 2017

@author: root
"""

import torch
import torch.nn as nn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np



import SADM as SDM

from utilsdm import load_pairs_csv,imreadtonumpy,fast_hist,get_NMM,get_MCC
import math
import fnmatch  

from collections import OrderedDict

from proposals_utils import *

import crf_inference as crfi


"""
deepmask import
"""
from collections import namedtuple
import sys
sys.path.append(os.path.join(os.getcwd(),'deepmask'))
import models
from PIL import Image
from tools.InferDeepMask import Infer
from utils.load_helper import load_pretrain


"""
superglue import
"""
sys.path.append(os.path.join(os.getcwd(),'superglue'))
from sgmodels.superpoint import SuperPoint
from sgmodels.superglue import SuperGlue
"""
from sgmodels.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,frame2tensor,process_resize,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
"""

torch.set_grad_enabled(False)

def get_f1score(hist,gt):
    gt_size = float(np.sum(gt))
    tp = float(hist[1,1])
    fn = float(hist[1,0])
    fp = float(hist[0,1])
    if gt_size == 0 or tp ==0:
        return 0,0,0
    if tp+fp > 0:
        precision = tp/(tp+fp)
    else:
        precision = 0
    recall = tp/(tp+fn)
    f1_score = (2.0*precision*recall)/(precision+recall)
    return precision,recall,f1_score
    
def range_end(start, stop, step=1):
    return np.arange(start, stop+step, step)

class Test_COCO(object):
    def __init__(self, args):
        
        """
        data prepare
        """
        self.data_path = args.data_path
        self.list_path = args.list_path
        self.test_start = args.test_start
        self.test_num = args.test_num
        pair_list_full = load_pairs_csv(os.path.join(self.data_path,self.list_path))
        if len(pair_list_full) < self.test_start + self.test_num:
            print('The test number is larger than the data length!')
            return
        self.pair_list = pair_list_full[self.test_start:(self.test_start + self.test_num)]
        """
        model initialization
        """
        self.model_path = args.model_path
        
        self.gpu = args.gpu_idx
        self.nolabel = args.nolabel
        self.input_scale = args.input_scale
        self.sort_num = args.sort_num
        
        
        self.loc = SDM.SelfDM_VGG(self.nolabel, self.sort_num)
        loc_saved_state_dict = torch.load(self.model_path)
        state_dict = loc_saved_state_dict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name]=v
        self.loc.load_state_dict(new_state_dict)
        self.loc.cuda(self.gpu)
        
        """
        visualization options
        """
        self.vis_loc_flag = args.vis_loc_flag
        self.vis_loc_path = args.vis_loc_path
            
        self.vis_deepmask_flag = args.vis_deepmask_flag
        self.vis_deepmask_path = args.vis_deepmask_path
        
        self.vis_final_flag = args.vis_final_flag
        self.vis_final_path = args.vis_final_path
        
        self.sigmoid_mod = nn.Sigmoid()
        
        self.vis_deepmask_flag = args.vis_deepmask_flag
        """
        deepmask
        """
        Config = namedtuple('Config', ['iSz', 'oSz', 'gSz', 'batch'])
        config = Config(iSz=160, oSz=56, gSz=112, batch=1)  # default for training
        self.deepmask_model = (models.__dict__[args.arch](config))
        self.deepmask_model = load_pretrain(self.deepmask_model, args.resume)
        

        self.scales = [2**i for i in range_end(args.si, args.sf, args.ss)]
        self.meanstd = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        self.nps = args.nps
        self.infer = Infer(nps=self.nps, scales=self.scales, meanstd=self.meanstd, model=self.deepmask_model, device=self.gpu)
        
        
        """
        superglue
        """
        self.resize_max = args.resize_max
        self.resize_min = args.resize_min
        
        config = {
            'superpoint': {
                'nms_radius': args.nms_radius,
                'keypoint_threshold': args.keypoint_threshold,
                'max_keypoints': args.max_keypoints
            },
            'superglue': {
                'weights': args.superglue,
                'sinkhorn_iterations': args.sinkhorn_iterations,
                'match_threshold': args.match_threshold,
            }
        }
        
        self.resize_float = args.resize_float
        
        self.superpoint = SuperPoint(config.get('superpoint', {})).to(self.gpu)
        self.superglue = SuperGlue(config.get('superglue', {})).to(self.gpu)
        
        self.crf_flag = args.crf_flag
        self.softmax_mask = SoftmaxMask()
        
        self.iou = 0
        self.NMM = 0
        self.MCC = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        self.iou_p2 = 0
        self.NMM_p2 = 0
        self.MCC_p2 = 0
        self.precision_p2 = 0
        self.recall_p2 = 0
        self.f1_score_p2 = 0
    
    def test(self):
        self.loc.eval()
        self.superpoint.eval()
        self.superglue.eval()
        self.deepmask_model = self.deepmask_model.eval().to(self.gpu)
        
        img_count_fore = 0
        detect_count = 0
        for piece in self.pair_list:
            img,_ = imreadtonumpy(self.data_path,piece[0],self.input_scale)
            gt_temp = cv2.imread(os.path.join(self.data_path,piece[1]))[:,:,0]
            gt_temp[gt_temp == 255] = 1
            gt_temp[gt_temp == 128] = 1
            gt_temp = cv2.resize(gt_temp,(self.input_scale,self.input_scale) , interpolation = cv2.INTER_NEAREST)
            gt = gt_temp
            image = torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float().cuda(self.gpu)
            output_a = self.loc(image)
            output_b = self.sigmoid_mod(output_a[0])
            output = output_b/float(1.0)
                
            if self.vis_loc_flag == True:
                self.vis_fun(f_name,output)
            im = np.array(Image.open(os.path.join(self.data_path,piece[0])).convert('RGB'), dtype=np.float32)
            h, w = im.shape[:2]
            img_deepmask = np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0).astype(np.float32)
            img_deepmask = torch.from_numpy(img_deepmask / 255.).to(self.gpu)
            self.infer.forward(img_deepmask)
            masks, scores = self.infer.getTopProps(.2, h, w)
        
            scoremap = output.cpu().data[0].numpy()
            h_sm, w_sm = scoremap.shape[:2]
            if h_sm != h or w_sm != w:
                scoremap = cv2.resize(scoremap, (w, h))
        
            selected_box = maskselection(masks,scoremap)
                
            box_resize = fixsize(selected_box,self.resize_max,self.resize_min)
        
            final_score,all_mkpts = get_finallabel(im,scoremap,selected_box,self.gpu,box_resize, self.resize_float, self.superpoint, self.superglue)
            
            piece_0 = piece[0].split('/')
            p_0_len = len(piece_0)
            piece_0_ = piece_0[p_0_len-1].split('.')
            pathname = os.path.join('vis_coco20ps', piece_0_[0]+'.png') 
            outputp = np.uint8(final_score * 255)
            maskp = cv2.applyColorMap(outputp,cv2.COLORMAP_JET)
            cv2.imwrite(pathname, maskp)
                
            if self.crf_flag:
                final_score_tensor = torch.from_numpy(final_score[np.newaxis, :]).float().cuda(self.gpu)
                    
                neg_final_score_tensor = final_score_tensor.clone()
                neg_final_score_tensor = 1.0 - final_score_tensor
                unary = torch.cat((neg_final_score_tensor,final_score_tensor),0) 
                unary = unary.cpu().data.numpy()
                shape = unary.shape[0:3]
                unary.reshape([1, shape[0], shape[1], shape[2]])
                unary = unary[np.newaxis, :]
                    #print unary.shape[0:4]
                img = img.copy()
                img[:,:,0] = img[:,:,0] + 104.008
                img[:,:,1] = img[:,:,1] + 116.669
                img[:,:,2] = img[:,:,2] + 122.675
                img = img[np.newaxis, :].transpose(0,3,1,2)
                prediction = crfi.do_crf_inference(img,unary)
                prediction = 1.0 - self.softmax_mask(prediction)
                
                prediction = prediction.cpu().data[0].numpy()
                prediction = prediction[:self.input_scale,:self.input_scale]
            else:
              continue
#        
            o_tmp = prediction
            
            piece_0 = piece[0].split('/')
            p_0_len = len(piece_0)
            piece_0_ = piece_0[p_0_len-1].split('.')
            pathname = os.path.join('vis_coco20pscrf', piece_0_[0]+'.png') 
            outputp = np.uint8(prediction * 255)
            maskp = cv2.applyColorMap(outputp,cv2.COLORMAP_JET)
            cv2.imwrite(pathname, maskp)
            
            o_perc = o_tmp[o_tmp > 0.5].mean()
            prediction[prediction > 0.5] = 1
            prediction[prediction <= 0.5] = 0
            output = prediction.astype(int)
            
            hist = fast_hist(gt.flatten(),output.flatten(),self.nolabel+1).astype(float)
        
            NMM_tmp = get_NMM(hist,gt)
            self.NMM += NMM_tmp
            
            MCC_tmp = get_MCC(hist)
            self.MCC += MCC_tmp
        
            iou_tmp = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
            self.iou += iou_tmp[1]
            
            p_tmp,r_tmp,f1_tmp = get_f1score(hist,gt)
            self.precision += p_tmp
            self.recall += r_tmp
            self.f1_score += f1_tmp
            
            
            if math.isnan(o_perc):
                o_perc = 0.0
            
            if o_perc > 0.5:
                detect_count += 1
                self.NMM_p2 += NMM_tmp
                self.MCC_p2 += MCC_tmp
                self.iou_p2 += iou_tmp[1]
                self.precision_p2 += p_tmp
                self.recall_p2 += r_tmp
                self.f1_score_p2 += f1_tmp
            
            
            
            print('item ' + str(img_count_fore) + ' processed!')
            img_count_fore += 1
            
            
            
            
            
            
            
            final_label = prediction * 255
                
            if self.vis_final_flag == True: 
                piece_0 = piece[0].split('/')
                p_0_len = len(piece_0)
                piece_0_ = piece_0[p_0_len-1].split('.')
                cv2.imwrite(os.path.join(self.vis_final_path,piece_0_[0]+'.png'),final_label)            
                            
            if self.vis_deepmask_flag:
                self.vis_boxes(selected_box,im,scoremap,all_mkpts,f_name)    
                    
        self.NMM = self.NMM/img_count_fore
        self.MCC = self.MCC/img_count_fore
        self.iou = self.iou/img_count_fore
        self.precision = self.precision/img_count_fore  
        self.recall = self.recall/img_count_fore 
        self.f1score = self.f1_score/img_count_fore
        
        self.NMM_p2 = self.NMM_p2/detect_count    
        self.MCC_p2 = self.MCC_p2/detect_count   
        self.iou_p2 = self.iou_p2/detect_count
        self.precision_p2 = self.precision_p2/detect_count    
        self.recall_p2 = self.recall_p2/detect_count
        self.f1score_p2 = self.f1_score_p2/detect_count
        
        print('Protocol 1  iou  avg  = ',self.iou)
        print('Protocol 1  NMM  avg  = ',self.NMM)
        print('Protocol 1  MCC  avg  = ',self.MCC)
        print('Protocol 1  precision  avg  = ',self.precision)
        print('Protocol 1  recall  avg  = ',self.recall)
        print('Protocol 1  f1score  avg  = ',self.f1score)
        print('Protocol 2  iou  avg  = ',self.iou_p2)
        print('Protocol 2  NMM  avg  = ',self.NMM_p2)
        print('Protocol 2  MCC  avg  = ',self.MCC_p2)
        print('Protocol 2  precision  avg  = ',self.precision_p2)
        print('Protocol 2  recall  avg  = ',self.recall_p2)
        print('Protocol 2  f1score  avg  = ',self.f1score_p2)
                
                
    def vis_boxes(self,boxes,im,scoremap,points_list,f_name):
        #print(len(boxes))
        res = im[:,:,::-1].copy().astype(np.uint8)
        scoremap[scoremap > 0.5] = 1
        scoremap[scoremap <= 0.5] = 0
        res[:, :, 0] = scoremap[:, :] * 255 + (1 - scoremap[:, :]) * res[:, :, 0]
        for i in range(len(boxes)):
            res = cv2.rectangle(res, (boxes[i][0][0], boxes[i][0][1]),
                        (boxes[i][0][2], boxes[i][0][3]), (0, 255, 0), 2)
        point_size = 1
        point_color = (0, 0, 255) # BGR
        thickness = 4    
        for point in points_list:
            cv2.circle(res, point, point_size, point_color, thickness)
        strname = os.path.join(self.vis_deepmask_path,f_name[:-4]+'_proposal.png')
        cv2.imwrite(strname,res)    
        
    def vis_masks(self,masks,im,scoremap):
        res = im[:,:,::-1].copy().astype(np.uint8)
        #res[:, :, 2] = masks[:, :, i] * 255 + (1 - masks[:, :, i]) * res[:, :, 2]
        for i in range(masks.shape[2]):
            mask = masks[:, :, i].astype(np.uint8)
            _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_area = [cv2.contourArea(cnt) for cnt in contours]
            cnt_max_id = np.argmax(cnt_area)
            contour = contours[cnt_max_id]
            polygons = contour.reshape(-1, 2)

            predict_box = cv2.boundingRect(polygons)

            res = cv2.rectangle(res, (predict_box[0], predict_box[1]),
                        (predict_box[0]+predict_box[2], predict_box[1]+predict_box[3]), (0, 255, 0), 3)
        strname = os.path.join(self.vis_deepmask_path,f_name[:-4]+'_proposal.png')
        cv2.imwrite(strname,res)        
            
    
                   
        
    def vis_fun(self,output):
        output = self.sigmoid_mod(output[0]).cpu().data[0].numpy()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        output = np.uint8(output * 255)
        cv2.imwrite(self.vis_loc_path, output)
        
        
        
class SoftmaxMask(nn.Module):
    def __init__(self):
        super(SoftmaxMask,self).__init__()
        self.softmax = nn.Softmax2d()
        
    def forward(self,x):
        x = self.softmax(x)
        return x[:,0,:,:]        
        



        
"""
The parameters for testing.
"""
class ArgsLocal:
    pass

args = ArgsLocal()


args.model_path = 'selfdm.pth'

args.gpu_idx = 0
args.nolabel = 1
args.sort_num = 48
args.input_scale = 512

args.vis_loc_flag = False
args.vis_loc_path = 'como_deepmask'
args.data_path =  'BESTI/test/'
args.list_path = args.data_path +  'test.csv'
args.test_start = 0
args.test_num = 1000

args.vis_deepmask_flag = False
args.vis_deepmask_path = 'vis_CoMoFoD/como_deepmask'

args.arch = 'DeepMask'
args.resume = 'deepmask/pretrained/deepmask/DeepMask.pth.tar'
args.si=-2.5#initial scale
args.sf=0.5#final scale
args.ss=0.5#scale step
args.nps = 500 #number of proposals to save in test

args.vis_final_flag = False
args.vis_final_path = 'vis_coco/SDMPSCRF/'


"""
superglue
"""
args.nms_radius = 4
args.keypoint_threshold = 0.000 # 0.005
args.max_keypoints = 2048# 1024
args.superglue = 'indoor' # choices={'indoor', 'outdoor'}
args.sinkhorn_iterations = 20
args.match_threshold = 0.4 
args.resize_max = [2000, 1500]
args.resize_min = [320, 240]
args.resize_float = True

args.crf_flag = True

tc = Test_COCO(args)
tc.test()
