# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:22:26 2023

@author: abed
"""
import pickle
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd
import glob
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import pickle
import joblib
import time
import gzip
import os


def pav(y):

    y = np.asarray(y)
    #assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    
    
    lvlsets = np.c_[lvls, lvls]
    #print(lvlsets)
    flag = 1
    while flag:
        deriv = np.diff(v)
        
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    return v

def tippet_plot(LR_scores, LR_scores_log, same_spk_list):
	
  
    LR_same = np.array(LR_scores_log[same_spk_list == 1])
    LR_diff = np.array(LR_scores_log[same_spk_list != 1])

    
    print("LR_minmax")
    LR_min = np.min([np.min(np.array(LR_same)),np.min(np.array(LR_diff))])
    LR_max = np.max([np.max(np.array(LR_same)),np.max(np.array(LR_diff))])
    tippetX = np.arange(LR_min,LR_max,(LR_max-LR_min)/100)
    print("tippet data")
    tippet_LR_same = [len(LR_same[LR_same<i])/len(LR_same) for i in tippetX]
    tippet_LR_diff = [len(LR_diff[LR_diff>=i])/len(LR_diff) for i in tippetX]
    print("drawing")
    plt.figure()
    plt.plot(tippetX,tippet_LR_diff,'--',label='different')
    plt.plot(tippetX,tippet_LR_same,label='same')
    plt.legend(fontsize='large')
    plt.ylabel('proportion', fontsize='large')
    plt.xlabel('logLR',fontsize='large')

    LR_same = np.array(LR_scores[same_spk_list == 1])
    LR_diff = np.array(LR_scores[same_spk_list != 1])
    
    print("calculating cllr")
    cllr = 0.0
    cllr_s = 0.0
    for i in range(len(LR_same)):
        cllr_s = cllr_s + np.log2(1+1/np.max([LR_same[i],0.000001]))
    cllr_s = cllr_s/len(LR_same)
    cllr_d = 0.0
    for i in range(len(LR_diff)):
        cllr_d = cllr_d + np.log2(1+LR_diff[i])            
    cllr_d = cllr_d/len(LR_diff)
    cllr = (cllr_s+cllr_d)/2
    print(cllr)
    
   # calculate cllr_min
    print("calculating cllr min")
    
    X = list(LR_scores)
    
    LR_scores_sort_index = np.argsort(X)
    X_sorted = np.take_along_axis(np.array(X), np.array(LR_scores_sort_index), axis=0)
    y_sorted = np.take_along_axis(np.array(same_spk_list), np.array(LR_scores_sort_index), axis=0)
    # X_sorted=X_sorted.reshape(-1,)
    # y_sorted=y_sorted.reshape(-1,)
    
    pavm = IsotonicRegression(out_of_bounds="clip").fit(X_sorted,y_sorted)
    
    #LR_scores_PAV = pav(X)
        
    LR_scores_PAV = pavm.predict(X)
    posterior_odds = np.divide(LR_scores_PAV,1-LR_scores_PAV)   
    prior_odds = len(same_spk_list[same_spk_list==1])/len(same_spk_list[same_spk_list!=1])
    llrs = posterior_odds / prior_odds
    # llrs=llrs.reshape(-1,1)
    
    LR_same = np.array(llrs[(same_spk_list == 1)])
    LR_diff = np.array(llrs[(same_spk_list != 1)])
    cllr_min = 0.0
    cllr_s = 0.0
    for i in range(len(LR_same)):
        cllr_s = cllr_s + np.log2(1+1/np.max([LR_same[i],0.000001]))
    cllr_s = cllr_s/len(LR_same)
    cllr_d = 0.0
    for i in range(len(LR_diff)):
        cllr_d = cllr_d + np.log2(1+LR_diff[i])
    cllr_d = cllr_d/len(LR_diff)
    cllr_min = (cllr_s+cllr_d)/2
    print(cllr_min)
    return cllr,cllr_min,cllr-cllr_min,plt,cllr_d

def evl(csv_zip_path,out_path):
    
    csv_f=csv_zip_path  # csv file path
    out_re=out_path # output path 
    
    file_name = os.path.basename(csv_f)
    file_save_name=file_name+'.png'
    full_file_path = os.path.join(out_re, file_save_name)
    
    # Open the gzipped CSV file
    #with gzip.open(csv_f, 'rt', encoding='cp1252') as gzfile:
     #   df_pairs = pd.read_csv(gzfile, sep=",")
    df_pairs = pd.read_csv(csv_f, sep=",", encoding='cp1252')

    
    df_pairs["target_num"] = [1 if x else 0 for x in df_pairs["same or different speaker"] == 1]

    cllr,cllr_min,cllr_cal,plt,cllr_d = tippet_plot(df_pairs["LR_score"],df_pairs["LR_score_log"],df_pairs["target_num"])
    print(f"{cllr} {cllr_min} {cllr_cal} {cllr_d}")
    fpr, tpr, thresholds = roc_curve(df_pairs["target_num"], df_pairs["LR_score_log"], pos_label=1)
    
      
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)   
    print(f"{eer} {thresh}")
    plt.title(f"cllr:{str(np.round(cllr,3))} cllr_cal:{(np.round(cllr_cal,3))} cllr_min:{str(np.round(cllr_min,3))} eer:{str(np.round(eer,3))}")
    plt.savefig(full_file_path,dpi=300)

def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == "__main__":
    directory = "C:\\Users\\R I B\\Desktop\\Study\\Project Laboratory\\Project_Laboratory\\files_for_eval"
    all_files = get_all_file_paths(directory)
    
    print("Evaluation of test samples")
    print("Start - ECAPA")
    start_time = time.time()

    # Define the output directory for evaluation results
    out_path = "C:\\Users\\R I B\\Desktop\\Study\\Project Laboratory\\Project_Laboratory\\evaluation_results"
        
    for f in all_files:
        evl(f,out_path)
        end_time = time.time()
        # Calculate the running time in seconds
        running_time = end_time - start_time

        print(f"Running time xvector: {running_time:.6f} seconds")
    

