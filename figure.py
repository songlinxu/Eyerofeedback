import time, os, math 
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd 
import pingouin as pg
from scipy.signal import detrend
import seaborn as sns 
from EyeTrackingMetrics.eyesmetriccalculator import EyesMetricCalculator
from EyeTrackingMetrics.transition_matrix import *

from utils import visual_heatmap_individual, visual_radar_score, visual_gaze, plot_bar, generate_block_result
from utils import generate_avg_results, generate_gaze_norm_file, plot_box, generate_trial_results, generate_gaze_extension, visual_heatmap_subjective_questionnaire
from utils import visual_metric_curve, calculate_rm_corr, calculate_anova, calculate_anova_write



############################################################################################################
# Step 1: Dataset processing before drawing figures: Run codes below one by one to generate all result files for analysis.
# Note: You can also skip this step since we have generated all result files for figure drawing
############################################################################################################
# generate_gaze_norm_file('dataset/gaze_all.csv','dataset/gaze_all_norm.csv')
# generate_gaze_extension('dataset/gaze_all_norm.csv','dataset/gaze_all_norm_ext.csv')
# generate_trial_results('dataset/user_all_trial.csv','dataset/gaze_all_norm_ext.csv','dataset/user_all_trial_full.csv')
# generate_avg_results('dataset/user_all_trial_full.csv','dataset/gaze_all_norm_ext.csv','dataset/user_avg_norm.csv')
# generate_block_result('dataset/gaze_all_norm_ext.csv','dataset/gaze_block_2.5.csv',block_dur=2.5)


############################################################################################################
# Step 2: Main figure
############################################################################################################

# ANOVA analysis
# aov_data = pd.read_csv('dataset/user_avg_norm.csv')
# calculate_anova_write(aov_data,'resptime_norm')    
# calculate_anova_write(aov_data,'missnumber_norm')    
# calculate_anova_write(aov_data,'accuracy_norm')    
# calculate_anova_write(aov_data,'gaze_entropy_limit_norm')    
# calculate_anova_write(aov_data,'center_distance_norm')    
# calculate_anova_write(aov_data,'q1_norm')    
# calculate_anova_write(aov_data,'q2_norm')    
# calculate_anova_write(aov_data,'q3_norm')    
# calculate_anova_write(aov_data,'q4_norm')    
# calculate_anova_write(aov_data,'q5_norm')    
# calculate_anova_write(aov_data,'q6_norm')    

# box plot
# plot_box('dataset/user_avg_norm.csv',datatype = 'resptime_norm')
# plot_box('dataset/user_avg_norm.csv',datatype = 'gaze_entropy_limit_norm')
# plot_box('dataset/user_avg_norm.csv',datatype = 'center_distance_norm')


# questionnaire result
# visual_radar_score('dataset/user_avg_norm.csv','short','with distraction','_norm')
# visual_radar_score('dataset/user_avg_norm.csv','short','no distraction','_norm')
# visual_radar_score('dataset/user_avg_norm.csv','long','with distraction','_norm')
# visual_radar_score('dataset/user_avg_norm.csv','long','no distraction','_norm')


# headmap 
# visual_heatmap_individual('dataset/user_avg_norm.csv',None,'gaze_entropy_limit_norm')
# visual_heatmap_individual('dataset/user_avg_norm.csv',None,'center_distance_norm')

# gaze distribution: gaze data has been resampled to 1 Hz for visualization
# visual_gaze('dataset/gaze_all_norm.csv',18,tail='limit_norm')

# trend curve (trial, block, session)
# visual_metric_curve('dataset/gaze_block_2.5.csv',datatype = 'gaze_entropy_limit_norm',unit='block_id',dur_interval=2.5)
# visual_metric_curve('dataset/gaze_block_2.5.csv',datatype = 'gaze_entropy_limit_norm',unit='block_trial_id',dur_interval=2.5)

# visual_metric_curve('dataset/gaze_block_2.5.csv',datatype = 'center_distance_norm',unit='block_id',dur_interval=2.5)
# visual_metric_curve('dataset/gaze_block_2.5.csv',datatype = 'center_distance_norm',unit='block_trial_id',dur_interval=2.5)


# correlation figure
# calculate_rm_corr('dataset/user_all_trial_full.csv','resptime_norm','gaze_entropy_limit_norm')
# calculate_rm_corr('dataset/user_all_trial_full.csv','resptime_norm','center_distance_norm')


############################################################################################################
# Step 3: Appendix figure
############################################################################################################


# ANOVA analysis
# aov_data = pd.read_csv('dataset/user_avg_norm.csv')
# calculate_anova_write(aov_data,'resptime')    
# calculate_anova_write(aov_data,'missnumber')    
# calculate_anova_write(aov_data,'accuracy')    
# calculate_anova_write(aov_data,'gaze_entropy_limit')    
# calculate_anova_write(aov_data,'center_distance')    
# calculate_anova_write(aov_data,'q1')    
# calculate_anova_write(aov_data,'q2')    
# calculate_anova_write(aov_data,'q3')    
# calculate_anova_write(aov_data,'q4')    
# calculate_anova_write(aov_data,'q5')    
# calculate_anova_write(aov_data,'q6')    

# box plot
# plot_box('dataset/user_avg_norm.csv',datatype = 'resptime')
# plot_box('dataset/user_avg_norm.csv',datatype = 'gaze_entropy_limit')
# plot_box('dataset/user_avg_norm.csv',datatype = 'center_distance')


# questionnaire result
# visual_heatmap_subjective_questionnaire('dataset/user_avg_norm.csv')
# visual_radar_score('dataset/user_avg_norm.csv','short','with distraction','')
# visual_radar_score('dataset/user_avg_norm.csv','short','no distraction','')
# visual_radar_score('dataset/user_avg_norm.csv','long','with distraction','')
# visual_radar_score('dataset/user_avg_norm.csv','long','no distraction','')


# headmap 
# visual_heatmap_individual('dataset/user_avg_norm.csv',None,'gaze_entropy_limit')
# visual_heatmap_individual('dataset/user_avg_norm.csv',None,'center_distance')


# trend curve (trial, block, session)
# visual_metric_curve('dataset/gaze_block_2.5.csv',datatype = 'gaze_entropy_limit',unit='block_id',dur_interval=2.5)
# visual_metric_curve('dataset/gaze_block_2.5.csv',datatype = 'center_distance',unit='block_id',dur_interval=2.5)
# visual_metric_curve('dataset/gaze_block_2.5.csv',datatype = 'gaze_entropy_limit',unit='block_trial_id',dur_interval=2.5)
# visual_metric_curve('dataset/gaze_block_2.5.csv',datatype = 'center_distance',unit='block_trial_id',dur_interval=2.5)


# correlation figure
# calculate_rm_corr('dataset/user_all_trial_full.csv','resptime','gaze_entropy_limit')
# calculate_rm_corr('dataset/user_all_trial_full.csv','resptime','center_distance')


