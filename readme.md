# Self-regulating attention via synchronized bodily maps of oculomotor trajectory

This code is used to replicate the results, analysis and figures in the paper named "Self-regulating attention via synchronized bodily maps of oculomotor trajectory". 

## File Structure

- figure.py: data preprocess, analysis and figure plot.
- utils.py: functions used in figure.py
- dataset folder: raw dataset and processed dataset
- EyeTrackingMetrics folder: This folder comes from the open-source library (https://github.com/Husseinjd/EyeTrackingMetrics). We use it to calculate the gaze stationary entropy.

## Steps to Replicate the Results and Figures in the Paper

- Download gaze_all_norm_ext.csv from [here](https://drive.google.com/file/d/1dyKr52d0koPx_aMx0z9jIStWJTzewGfi/view?usp=sharing) and gaze_all_norm.csv from [here](https://drive.google.com/file/d/1besdlX_YSMztOnPWem0Or5Wuw9-4WHOr/view?usp=sharing) and gaze_all.csv from [here](https://drive.google.com/file/d/18OxJKmlnBO77B-TDfvZitxe2iQmWVdtF/view?usp=sharing) and put them into the dataset folder.

- Run the code block one by one in order to process the raw dataset and plot figures.






