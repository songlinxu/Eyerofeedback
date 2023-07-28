import time, os, math 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd 
import seaborn as sns 
from statsmodels.stats.anova import AnovaRM
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from collections import defaultdict
from EyeTrackingMetrics.eyesmetriccalculator import EyesMetricCalculator
from EyeTrackingMetrics.transition_matrix import *

from math import pi
from scipy.interpolate import make_interp_spline
from multiprocessing import Process, Array, Value, Queue
import multiprocessing


def calculate_gaze_entropy(gaze_x_arr,gaze_y_arr,screen_width,screen_height):
    # code adapted from https://github.com/Husseinjd/EyeTrackingMetrics
    TEST_SCREENDIM = [screen_width, screen_height]
    TEST_VERTICES = [[int(screen_width/4),int(screen_height/4)], [int(screen_width/4),int(screen_height/4*3)], [int(screen_width/4*3),int(screen_height/4*3)], [int(screen_width/4*3),int(screen_height/4)]]
    TEST_AOI_DICT = {'aoi_poly1': PolyAOI(TEST_SCREENDIM,TEST_VERTICES)}
    
    GAZE_ARRAY = np.concatenate((gaze_x_arr.reshape((len(gaze_x_arr),1)),gaze_y_arr.reshape((len(gaze_y_arr),1))),axis=1)
    GAZE_ARRAY = np.concatenate((GAZE_ARRAY,np.zeros((len(GAZE_ARRAY),1))),axis=1)

    ec = EyesMetricCalculator(None,GAZE_ARRAY,TEST_SCREENDIM)
        
    gaze_entropy = ec.GEntropy(TEST_AOI_DICT,'stationary').compute()
        
    return gaze_entropy

def func_table_norm(input_table,datatype_list):
    '''Note that this function will change the order of the table data'''
    print('Note that this function will change the order of the table data')
    user_id_list = list(set(input_table['user_id']))
    header = list(input_table.columns.values)
    for datatype in datatype_list:
        header.append(datatype+'_norm')
    for i,user_id in enumerate(user_id_list):
        user_table = input_table[input_table['user_id']==user_id]
        raw_arr = np.array(user_table)
        for j,datatype in enumerate(datatype_list):
            target_arr = np.array(user_table[datatype]).flatten()
            print(user_id,datatype)
            print(target_arr)
            arr_std = np.std(target_arr)
            arr_mean = np.mean(target_arr)
            if arr_std == 0:
                print('error! std is zero!!!','!'*20)
                print(user_id,datatype)
                print(target_arr)
                normed_arr = target_arr-arr_mean
            else:
                normed_arr = (target_arr-arr_mean)/arr_std
            normed_arr = normed_arr.reshape((len(normed_arr),1))
            if j==0:
                vertical_arr = normed_arr.copy()
            else:
                vertical_arr = np.concatenate((vertical_arr,normed_arr),axis=1)
        concat_arr = np.concatenate((raw_arr,vertical_arr),axis=1)
        if i==0:
            horiz_arr = concat_arr.copy()
        else:
            horiz_arr = np.concatenate((horiz_arr,concat_arr),axis=0)
    
    table_norm = pd.DataFrame(horiz_arr,columns=header)
    return table_norm


def generate_gaze_norm_file(input_file,output_file):
    user_gaze = pd.read_csv(input_file)
    screen_width_const = np.array(user_gaze['screen_width'])[-1]
    screen_height_const = np.array(user_gaze['screen_height'])[-1]
    gx = np.array(user_gaze['gaze_x_raw']).flatten()
    gy = np.array(user_gaze['gaze_y_raw']).flatten()

    gx_limit = func_limit_arr(gx,screen_width_const,0)
    gy_limit = func_limit_arr(gy,screen_height_const,0)
    gx_limit = gx_limit.reshape((len(gx_limit),1))
    gy_limit = gy_limit.reshape((len(gy_limit),1))

    user_gaze_new = pd.concat([user_gaze,pd.DataFrame(np.concatenate((gx_limit,gy_limit),axis=1),columns=['gaze_x_limit','gaze_y_limit'])],axis=1)

    user_gaze_whole = func_table_norm(user_gaze_new,['gaze_x_raw','gaze_y_raw','gaze_x_limit','gaze_y_limit'])
    user_gaze_whole.to_csv(output_file,index=False)


def generate_avg_results(input_result_file,input_gaze_file,output_file):
    user_result = pd.read_csv(input_result_file)
    user_gaze = pd.read_csv(input_gaze_file)
    user_avg_list = []

    user_id_list = list(set(user_result['user_id']))

    for i,user_id in enumerate(user_id_list):
        for j,duration in enumerate(['short','long']):
            for k,distraction in enumerate(['with distraction','no distraction']):
                for h,threshold in enumerate(['none','static','filtered']):
                    item_data = user_result[(user_result['duration_text']==duration)&(user_result['condition_text']==distraction)&(user_result['threshold_text']==threshold)&(user_result['user_id']==user_id)]
                    item_gaze = user_gaze[(user_gaze['duration_text']==duration)&(user_gaze['condition_text']==distraction)&(user_gaze['threshold_text']==threshold)&(user_gaze['user_id']==user_id)]
                    print(user_id,duration,distraction,threshold,len(item_data),len(item_gaze))
                    acc = np.mean(np.array(item_data['accuracy']))
                    resptime = np.mean(np.array(item_data['resptime'])) 
                    missnumber = 10-len(item_data)
                    screen_width_const = np.array(item_gaze['screen_width'])[-1]
                    screen_height_const = np.array(item_gaze['screen_height'])[-1]
                    gx = np.array(item_gaze['gaze_x_raw']).flatten()
                    gy = np.array(item_gaze['gaze_y_raw']).flatten()
                    gx_raw = gx.copy()
                    gy_raw = gy.copy()
                    gx_raw = gx_raw.reshape((len(gx_raw),1))
                    gy_raw = gy_raw.reshape((len(gy_raw),1))
                    gx_limit = func_limit_arr(gx,screen_width_const,0)
                    gy_limit = func_limit_arr(gy,screen_height_const,0)
                    gx_limit = gx_limit.reshape((len(gx_limit),1))
                    gy_limit = gy_limit.reshape((len(gy_limit),1))
                    gaze_entropy_raw = calculate_gaze_entropy(gx_raw,gy_raw,screen_width_const,screen_height_const)
                    gaze_entropy_limit = calculate_gaze_entropy(gx_limit,gy_limit,screen_width_const,screen_height_const)
                    q1 = np.mean(np.array(item_data['q1']))
                    q2 = np.mean(np.array(item_data['q2']))
                    q3 = np.mean(np.array(item_data['q3']))
                    q4 = np.mean(np.array(item_data['q4']))
                    q5 = np.mean(np.array(item_data['q5']))
                    q6 = np.mean(np.array(item_data['q6']))
                    user_avg_list.append([user_id,duration,distraction,threshold,acc,resptime,missnumber,gaze_entropy_raw,gaze_entropy_limit,q1,q2,q3,q4,q5,q6])


    header = ['user_id','duration_text','condition_text','threshold_text','accuracy','resptime','missnumber','gaze_entropy_raw','gaze_entropy_limit','q1','q2','q3','q4','q5','q6']
    user_avg_arr = np.array(user_avg_list)
    user_avg_table = pd.DataFrame(user_avg_arr,columns=header)
    user_avg_table.to_csv(output_file,index=False)

def generate_avg_norm_file(input_file,output_file):
    user_avg_table = pd.read_csv(input_file)
    user_gaze_whole = func_table_norm(user_avg_table,['accuracy','resptime','missnumber','gaze_entropy_raw','gaze_entropy_limit','q1','q2','q3','q4','q5','q6'])
    user_gaze_whole.to_csv(output_file,index=False)


def find_nearest(array, value):     
    array = np.asarray(array)    
    idx = (np.abs(array - value)).argmin()  
    return idx



def visual_distribution_all(table_file,user_id_list,datatype):
    user_table_all = pd.read_csv(table_file)
    user_table_select = pd.DataFrame(columns=user_table_all.columns.values)
    if user_id_list != None:
        for i in user_id_list:
            user_item = user_table_all[(user_table_all['user_id']==i)]
            user_table_select = pd.concat([user_table_select,user_item],axis=0)
    else:
        user_table_select = user_table_all
    fig, axes = plt.subplots(1,1,figsize = (10,10))
    sns.displot(data=user_table_select,x=datatype,hue='threshold_text',kind='kde',linewidth=4)
    
    plt.show()

def visual_heatmap_individual(table_file,user_id_list,datatype):
    user_table_all = pd.read_csv(table_file)
    user_table_select = pd.DataFrame(columns=user_table_all.columns.values)
    if user_id_list != None:
        for i in user_id_list:
            user_item = user_table_all[(user_table_all['user_id']==i)]
            user_table_select = pd.concat([user_table_select,user_item],axis=0)
    else:
        user_table_select = user_table_all
    fig, axes = plt.subplots(1,1,figsize = (5,5))
    matrix_data = np.empty((0,21))
    for i,duration in enumerate(['short','long']):
        for j,distraction in enumerate(['with distraction', 'no distraction']):
            line_data = np.empty((3,0))
            for k,threshold in enumerate(['none','static','filtered']):
                result_data_select = user_table_select[(user_table_select['duration_text']==duration)&(user_table_select['condition_text']==distraction)&(user_table_select['threshold_text']==threshold)]
                line_data = np.concatenate((line_data,np.array(result_data_select[datatype]).reshape((3,7))),axis=1)
            matrix_data = np.concatenate((matrix_data,line_data),axis=0)
                    
    print(matrix_data.shape)
    sns.heatmap(data = np.array(matrix_data).T)
    plt.show()

def visual_radar_score(data_csv,duration,distraction):
    data = pd.read_csv(data_csv)
    if duration != None:
        data = data[(data['duration_text']==duration)]
    if distraction != None:
        data = data[(data['condition_text']==distraction)]
    table_list = []
    for i,threshold in enumerate(['none','static','filtered']):
        data_item = data[data['threshold_text']==threshold]
        data_arr = np.array(data_item)
        data_arr = data_arr[:,-6:]
        table_list.append([threshold]+list(np.mean(data_arr,axis=0)))

    table = pd.DataFrame(np.array(table_list),columns=['threshold_text','q1','q2','q3','q4','q5','q6'])
    print(table)
    color_dict = {'none':'r','static':'g','filtered':'b'}
    visual_radar(table,hue='threshold_text',color_dict=color_dict,step=3)


def visual_radar(data,hue,color_dict,step = 3):
    # code adapted from https://python-graph-gallery.com/391-radar-chart-with-several-individuals/
    group_list = list(set(data[hue]))
    categories=list(data)[1:]
    N = len(categories)

    data_value = np.float_(np.array(data)[:,1:])
    
    max_value = np.max(data_value)
    min_value = np.min(data_value)
    print('max: ', max_value, ' min: ', min_value)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    ax = plt.subplot(111, polar=True)
 
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    plt.xticks(angles[:-1], categories)
 
    ax.set_rlabel_position(0)

    max_limit = max_value*1.1 if max_value > 0 else max_value*0.9
    min_limit = min_value*0.9 if min_value > 0 else min_value*1.1

    interval = (max_limit - min_limit)/step
    ytick_list = [min_limit + interval * (k+1) for k in range(step)]
    ytick_str = [str((min_limit+interval * (k+1))) for k in range(step)]
    print(ytick_list)
    print(ytick_str)

    plt.yticks(ytick_list, ytick_str, color="grey", size=7)
    # plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    plt.ylim(min_limit,max_limit)


    for i in range(len(group_list)):
        df_item = data[data[hue]==group_list[i]]
        values=list(np.float_(np.array(df_item)[0][1:]))
        values += values[:1]

        X_Y_Spline = make_interp_spline(angles[:-1], values[:-1])
        X_ = np.linspace(min(angles), max(angles), 500)
        Y_ = X_Y_Spline(X_)

        X_ = list(X_)
        Y_ = list(Y_)

        X_ += angles[:1]
        Y_ += values[:1]

        ax.plot(angles, values, linewidth=1, linestyle='solid', label="group " + str(i), c=color_dict[group_list[i]])
        ax.fill(angles, values, color_dict[group_list[i]], alpha=0.1)
 
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()



def visual_gaze(gaze_all_file,user_id,tail='raw'):
    # warning: this code function is only suitable for one user, not multiple users together. 
    gaze_data_all = pd.read_csv(gaze_all_file)
    color_dict = {'none':'b','filtered':'g','static':'r'}
    user_table_select = gaze_data_all[gaze_data_all['user_id']==user_id]
    fig, axes = plt.subplots(3,4,figsize = (18,10))
    for i,duration in enumerate(['short','long']):
        for j,distraction in enumerate(['with distraction', 'no distraction']):
            for k,feedback in enumerate(['none','static','filtered']):
                gaze_data_select = user_table_select[(user_table_select['duration_text']==duration)&(user_table_select['condition_text']==distraction)&(user_table_select['threshold_text']==feedback)]

                x_data = gaze_data_select['gaze_x_'+tail]
                y_data = gaze_data_select['gaze_y_'+tail]
                
                axes[k][i*2+j].scatter(x_data,y_data,s=4,c=color_dict[feedback])
                axes[k][i*2+j].set(facecolor = "black")
                axes[k][i*2+j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    plt.subplots_adjust(hspace = 0.03,wspace = 0.03)
    
    plt.show()




def plot_bar(data_csv,datatype = 'resptime'):
    user_table_select = pd.read_csv(data_csv)
    sns.barplot(user_table_select,x='threshold_text',y=datatype)
    sns.despine()
    plt.show()

def plot_box(data_csv,datatype = 'resptime'):
    label_dict = {'resptime_norm': 'Normalized Response Time', 'gaze_entropy_limit_norm': 'Normalized Gaze Entropy'}
    user_table_select = pd.read_csv(data_csv)

    fig, axes = plt.subplots(2,2,figsize = (6,6))
    for i,duration in enumerate(['short','long']):
        for j,distraction in enumerate(['with distraction', 'no distraction']):
            user_table_item = user_table_select[(user_table_select['duration_text']==duration)&(user_table_select['condition_text']==distraction)]
            sns.boxplot(user_table_item,x='threshold_text',y=datatype,ax=axes[i][j])
            axes[i][j].set_xlabel(duration+','+distraction)
            axes[i][j].set_ylabel(label_dict[datatype])
            axes[i][j].set_xticklabels(['Silence','Stationary','Filter'])
    sns.despine()
    plt.subplots_adjust(hspace = 0.5,wspace = 0.5,bottom=0.35)
    plt.show()




def func_limit_arr(input_arr, max_value, min_value):
    new_arr = []
    for arr_item in input_arr:
        if arr_item > max_value:
            new_arr.append(max_value)
        elif arr_item < min_value:
            new_arr.append(min_value)
        else:
            new_arr.append(arr_item)
    return np.array(new_arr)



