import time, os, math 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd 
import seaborn as sns 
from EyeTrackingMetrics.eyesmetriccalculator import EyesMetricCalculator
from EyeTrackingMetrics.transition_matrix import *

from utils import visual_distribution_all, visual_heatmap_individual, visual_radar_score, visual_gaze, plot_bar
from utils import generate_avg_results, generate_avg_norm_file, generate_gaze_norm_file, plot_box

############################################################################################################
# Dataset processing before drawing figures
############################################################################################################
# generate_gaze_norm_file('dataset/gaze_all.csv','dataset/gaze_all_norm.csv')
# generate_avg_results('dataset/user_all_trial.csv','dataset/gaze_all.csv','dataset/user_avg_raw.csv')
# generate_avg_norm_file('dataset/user_avg_raw.csv','dataset/user_avg_norm.csv')

############################################################################################################
# Main figure
############################################################################################################

# f2.b: bar plot for response time
# plot_bar('dataset/user_avg_norm.csv',datatype = 'resptime_norm')

# f2.c: distribution plot for response time
# visual_distribution_all('dataset/user_avg_norm.csv',None,'resptime_norm')


# f2.d: radar plot for questionnaire score 
# visual_radar_score('dataset/user_avg_norm.csv',None,None)

# f2.a: heatmap for gaze entropy
# visual_heatmap_individual('dataset/user_avg_norm.csv',None,'gaze_entropy_limit_norm')

# f2.e: bar plot for gaze entropy
# plot_bar('dataset/user_avg_norm.csv',datatype = 'gaze_entropy_limit_norm')

############################################################################################################
# Appendix figure
############################################################################################################


# f2.1: boxplot for resptime in 2x2 conditions
# plot_box('dataset/user_avg_norm.csv',datatype = 'resptime_norm')

# f2.2: radar plot for subjective scores in 2x2 conditions
# visual_radar_score('dataset/user_avg_norm.csv','short','with distraction')
# visual_radar_score('dataset/user_avg_norm.csv','short','no distraction')
# visual_radar_score('dataset/user_avg_norm.csv','long','with distraction')
# visual_radar_score('dataset/user_avg_norm.csv','long','no distraction')


# f2.3: boxplot for gaze entropy in 2x2 conditions
# plot_box('dataset/user_avg_norm.csv',datatype = 'gaze_entropy_limit_norm')


# f3: gaze distribution: gaze data has been resampled to 1 Hz for visualization
# visual_gaze('dataset/gaze_all_norm.csv',18,tail='limit_norm')