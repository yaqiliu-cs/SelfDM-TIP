#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sep. 2020

@author: liuyaqi
"""
import torch
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'superglue'))
from sgmodels.utils import (frame2tensor,process_resize)
import copy


"""
deepmask
"""
def compute_iou(rec1, rec2):
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0,0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0, intersect * 1.0
            
def compute_inter(rec1, rec2):
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        return (right_line - left_line) * (bottom_line - top_line) * 1.0
    
def maskselection(masks,scoremap,threshold_scorebox=[0.4,0.3,0.2], threshold_iou = 0.5,threshold_inter=0.8, area_ratio = 0.4):
    h, w = scoremap.shape
    size_whole = w * h
    final_maskbox = []
    for i in range(masks.shape[2]):
        mask = masks[:, :, i].astype(np.uint8)
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(cnt_area)==0:
            continue
        cnt_max_id = np.argmax(cnt_area)
        contour = contours[cnt_max_id]
        polygons = contour.reshape(-1, 2)
        predict_box = cv2.boundingRect(polygons)
        predict_box_size = predict_box[2] * predict_box[3]
        if predict_box_size > area_ratio * size_whole:
            continue
        predict_box_cordinate = [predict_box[0], predict_box[1],predict_box[0]+predict_box[2], predict_box[1]+predict_box[3]]
        cropped = scoremap[predict_box_cordinate[1]:predict_box_cordinate[3], predict_box_cordinate[0]:predict_box_cordinate[2]]
        cropped_score = np.mean(cropped)
        
        if cropped_score >= threshold_scorebox[0]:
            if len(final_maskbox) == 0:
                final_maskbox.append((predict_box_cordinate,cropped_score,predict_box_size))
                
            else:
                insert_flag = 0
                for i_fm in range(len(final_maskbox)):
                    iou_tmp, inter = compute_iou(final_maskbox[i_fm][0],predict_box_cordinate)
                    if iou_tmp >= threshold_iou:                       
                        if cropped_score >= final_maskbox[i_fm][1]:
                            final_maskbox[i_fm]=(predict_box_cordinate,cropped_score,predict_box_size)
                            
                            insert_flag = 1
                            break
                    
                    if inter/final_maskbox[i_fm][2] >= threshold_inter or inter/predict_box_size >= threshold_inter:
                        tmpbox = merge2boxes(predict_box_cordinate,final_maskbox[i_fm][0],scoremap)
                        
                        if tmpbox[2] <= area_ratio * size_whole and tmpbox[1] >= threshold_scorebox[1]:
                            final_maskbox[i_fm] = tmpbox
                            insert_flag = 1
                            break
                   
                if insert_flag == 0:
                    final_maskbox.append((predict_box_cordinate,cropped_score,predict_box_size))
    
    
    selected_maskbox = []
    mergeindex = []
    for i in range(len(final_maskbox)):
        if (i in mergeindex):
            continue
        
        tmpbox = copy.deepcopy(final_maskbox[i])
        mergeflag = 0
        for j in range(i+1,len(final_maskbox)):
            if j in mergeindex:
                continue            
            inter = compute_inter(tmpbox[0],final_maskbox[j][0])
            if inter/tmpbox[2] > 0.5 or inter/final_maskbox[j][2] > 0.5:
                tmpbox = merge2boxes(tmpbox[0],final_maskbox[j][0],scoremap)
                if tmpbox[2] <= area_ratio * size_whole and tmpbox[1] >= threshold_scorebox[2]:
                    mergeindex.append(i)
                    mergeindex.append(j)
                    selected_maskbox.append(tmpbox)
                    mergeflag = 1
        if mergeflag == 0:
            selected_maskbox.append(final_maskbox[i])
            mergeindex.append(i)        
                    
    inter_threshold = [0.5,0.4,0.3,0.25,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]    
    for itk in range(50):
        if itk > 4 and len(final_maskbox) == len(selected_maskbox):            
            break
        final_maskbox = selected_maskbox
        selected_maskbox = []
        mergeindex = []
        for i in range(len(final_maskbox)):
            if (i in mergeindex):
                continue
        
            tmpbox = copy.deepcopy(final_maskbox[i])
            mergeflag = 0
            for j in range(i+1,len(final_maskbox)):
                if j in mergeindex:
                    continue            
                inter = compute_inter(tmpbox[0],final_maskbox[j][0])
                if inter/tmpbox[2] > inter_threshold[itk] or inter/final_maskbox[j][2] > inter_threshold[itk]:
                    tmpbox = merge2boxes(tmpbox[0],final_maskbox[j][0],scoremap)
              
                    if tmpbox[2] <= area_ratio * size_whole and tmpbox[1] >= threshold_scorebox[2]:
                        mergeindex.append(i)
                        mergeindex.append(j)
                        selected_maskbox.append(tmpbox)
                        mergeflag = 1
                    if mergeflag == 0:
                        if inter/final_maskbox[i][2] > 0.5 or inter/final_maskbox[j][2] > 0.5:
                            if inter/final_maskbox[i][2] <= inter/final_maskbox[j][2]:
                                selected_maskbox.append(final_maskbox[i])
                                mergeindex.append(i)
                                mergeindex.append(j)
                                mergeflag = 1
                            else:
                                selected_maskbox.append(final_maskbox[j])
                                mergeindex.append(i)
                                mergeindex.append(j)
                                mergeflag = 1
            if mergeflag == 0:
                selected_maskbox.append(final_maskbox[i])
                mergeindex.append(i)                
    
                            
    return selected_maskbox

def merge2boxes(rec1,rec2,scoremap):
    left_line = min(rec1[0], rec2[0])
    right_line = max(rec1[2], rec2[2])
    top_line = min(rec1[1], rec2[1])
    bottom_line = max(rec1[3], rec2[3])
    newsize = (bottom_line - top_line)*(right_line - left_line)
    cropped = scoremap[top_line:bottom_line, left_line:right_line]
    cropped_score = np.mean(cropped)
    return ([left_line, top_line, right_line, bottom_line],cropped_score,newsize)
    
"""
superglue
"""
def resize_pos(x1,y1,src_size,tar_size,corner):
    w1=src_size[0]
    h1=src_size[1]
    w2=tar_size[0]
    h2=tar_size[1]
    y2=(h2/h1)*y1 + corner[1]
    x2=(w2/w1)*x1 + corner[0]
    return (int(x2),int(y2))
        
    
    
    
def read_image(image_o, device, resize, rotation, resize_float):
    image = cv2.cvtColor(image_o,cv2.COLOR_BGR2GRAY)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
            
    image_o = cv2.resize(image_o, (w_new, h_new))
        

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
           scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales,image_o
    
    
def get_finallabel(img,scoremap,selected_box,gpu,self_resize, self_resize_float, self_superpoint, self_superglue):
    final_label = np.zeros((scoremap.shape[0], scoremap.shape[1]))
    all_mkpts = []
    
    if len(selected_box) >= 2:
        pred = {}
        slic_labels = []
        boxes_scoremap = []
        for i in range(len(selected_box)):
            box = selected_box[i]
            cropped = img[box[0][1]:box[0][3], box[0][0]:box[0][2]].copy().astype(np.uint8)
            image0, inp0, scales0,image_o0 = read_image(cropped, gpu, self_resize, 0, self_resize_float)
            pred0 = self_superpoint({'image': inp0})                
            pred = {**pred, **{k+str(i): v for k, v in pred0.items()}}
            pred = {**pred, **{'image'+str(i): inp0}}
                
            slic = cv2.ximgproc.createSuperpixelSEEDS(image_o0.shape[1],image_o0.shape[0],image_o0.shape[2],400,15,3,5,True)
            slic.iterate(image_o0,10)
            slic_labels.append(slic.getLabels())
            boxes_scoremap.append(np.zeros((self_resize[1],self_resize[0])))
                        
        for k in pred:
            if isinstance(pred[k], (list, tuple)):
                pred[k] = torch.stack(pred[k])
                
        
        for i in range(len(selected_box)):
            for j in range(i+1,len(selected_box)):
                inputdata = {}
                inputdata = {**inputdata,**{'image0':pred['image'+str(i)],
                                'keypoints0':pred['keypoints'+str(i)],
                                'scores0':pred['scores'+str(i)],
                                'descriptors0':pred['descriptors'+str(i)]}}
                inputdata = {**inputdata,**{'image1':pred['image'+str(j)],
                                'keypoints1':pred['keypoints'+str(j)],
                                'scores1':pred['scores'+str(j)],
                                'descriptors1':pred['descriptors'+str(j)]}}
                    
                sg_match_results = {**inputdata, **self_superglue(inputdata)}
                    
                sg_match_results = {k: v[0].cpu().numpy() for k, v in sg_match_results.items()}
                    
                kpts0, kpts1 = sg_match_results['keypoints0'], sg_match_results['keypoints1']
                matches0, matches1 = sg_match_results['matches0'], sg_match_results['matches1']
                conf0, conf1 = sg_match_results['matching_scores0'], sg_match_results['matching_scores1']
                valid0 = matches0 > -1
                valid1 = matches1 > -1
                if True in valid0:
                    mkpts0 = kpts0[valid0]
                    mconf0 = conf0[valid0]
                    for mkpt_i in range(len(mkpts0)):
                        kpt = mkpts0[mkpt_i]
                        label = slic_labels[i][int(kpt[1]),int(kpt[0])] 
                        if label >= 0:
                            boxes_scoremap[i][slic_labels[i]==label] = mconf0[mkpt_i]
                            slic_labels[i][slic_labels[i]==label]=-10
                
                if  True in valid1: 
                    mkpts1 = kpts1[valid1]
                    mconf1 = conf1[valid1]
                    
                    for mkpt_i in range(len(mkpts1)):
                        kpt = mkpts1[mkpt_i]
                        label = slic_labels[j][int(kpt[1]),int(kpt[0])]
                        
                        if label >= 0:
                            
                            boxes_scoremap[j][slic_labels[j]==label] = mconf1[mkpt_i]
                            slic_labels[j][slic_labels[j]==label]=-10

                    for kpt in mkpts0:
                        all_mkpts.append(resize_pos(kpt[0],kpt[1],[self_resize[0],self_resize[1]],[selected_box[i][0][2]-selected_box[i][0][0],selected_box[i][0][3]-selected_box[i][0][1]],[selected_box[i][0][0],selected_box[i][0][1]]))
                    for kpt in mkpts1:
                        all_mkpts.append(resize_pos(kpt[0],kpt[1],[self_resize[0],self_resize[1]],[selected_box[j][0][2]-selected_box[j][0][0],selected_box[j][0][3]-selected_box[j][0][1]],[selected_box[j][0][0],selected_box[j][0][1]]))
            

            
        for i in range(len(selected_box)):
            cmregion = slic_labels[i] < 0
            if True in cmregion:
                
                boxes_scoremap[i] = cv2.resize(boxes_scoremap[i].astype('float32'),(int(selected_box[i][0][2]-selected_box[i][0][0]), int(selected_box[i][0][3]-selected_box[i][0][1])))
                
                final_label[selected_box[i][0][1]:selected_box[i][0][3], selected_box[i][0][0]:selected_box[i][0][2]] = boxes_scoremap[i] + final_label[selected_box[i][0][1]:selected_box[i][0][3], selected_box[i][0][0]:selected_box[i][0][2]]
                
        
        alpha = 4
        beta = 4
        gamma = - 2.0
        background_label = np.zeros((scoremap.shape[0], scoremap.shape[1]))
        for i in range(len(selected_box)):
            box = selected_box[i]
            cropped_kptscore = final_label[box[0][1]:box[0][3], box[0][0]:box[0][2]].copy()
            if True in (cropped_kptscore > 0):
                background_label[box[0][1]:box[0][3], box[0][0]:box[0][2]] = 1
                cropped_scoremap = scoremap[box[0][1]:box[0][3], box[0][0]:box[0][2]].copy()
                final_label[box[0][1]:box[0][3], box[0][0]:box[0][2]] = cropped_kptscore * alpha + cropped_scoremap * beta + gamma
                
        final_label = sigmoid_num(final_label) * background_label
    
    if len(selected_box) < 2 or len(all_mkpts) < 8:        
        final_label = scoremap
            

    return  final_label,all_mkpts
    
    
def fixsize(boxes,maxsize,minsize):
    final_size = copy.deepcopy(minsize)
    for i in range(len(boxes)):
        w = boxes[i][0][2]-boxes[i][0][0]
        h = boxes[i][0][3]-boxes[i][0][1]
        if w > final_size[0] and w < maxsize[0]:
            final_size[0] = w
        if h > final_size[1] and h < maxsize[1]:
            final_size[1] = h
        
        if w >= maxsize[0]:
            final_size[0] = maxsize[0]
        if h >= maxsize[1]:
            final_size[1] = maxsize[1]
        
    return final_size
            
        
def sigmoid_num(x):
    s = 1 / (1 + np.exp(-x))
    return s       
    
    
    
                