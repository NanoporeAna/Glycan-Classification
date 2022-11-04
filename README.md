# Glycan-Classification
**1.Glycan-Classification description**                               
                                                                                
 Glycan-Classification serves as a machine learning-based nanopore analysis platform 
 for the identification of different glycan types.                 
 The code contains three functions as follows:                                
                                                                               
 1).The raw data is analyzed for KL divergence.   	            
                                                                                
 2).The data is preprocessed to obtain the features required for training.                                   
                                                                                
 3).Model training and testing by the dataset(at './data/').                                        


 **2.Operating procedures**                             
    First, you need to ensure that './data/' has the target file(xx.xlsx) you need, then execute _'featuresample.py'_ to extract 
 the corresponding molecular features, and then select the corresponding machine learning model training in the model file to 
 obtain the model and results.
1).Data feature extraction
    Make sure the following parameters are what you want:
    `file_name = [
            'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Cel-DPE-6SL-28930 events',
             'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Lac-DPE-6SL-27696 events',
             'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events'
             ]`
   ` num = 30
    sample_len = 2500`

2)Model training and testing:
	Make sure that the feature file is obtained from your previous data preprocessing, directly execute and select the ML model you need


**3.Folder describing**

1).Folder "data"

This Fold includes three subfolders:"train_set","test_set","pre_set"
 --"3S3FL-MPB---6S3FL-MPB---6S2FL-MPB": 
 --"3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB":
 --"Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL": 缺少描述

2).Folder "model"

This folder is mainly used to store model files for training dataset.

3).Folder "models"
This folder is mainly used to store model files for loading exiting models

4).picture
Store some result pictures.

**4. soft dependencies** 
For the original code, the software uses the following main dependencies:
numpy
pandas
joblib
time
matplotlib
scipy
sklearn
tqdm
os
seaborn