# Glycan-Classification
## 1.Glycan-Classification description                              
                                                                                
 Glycan-Classification serves as a machine learning-based nanopore analysis platform 
 for the identification of different glycan types.                 
 The code contains three functions as follows:                                
                                                                               
 1).The raw data is analyzed for KL divergence.   	            
                                                                                
 2).The data is preprocessed to obtain the features required for training.                                   
                                                                                
 3).Model training and testing by the dataset(at './data/').                                        

---------
## 2.Operating procedures                           
    
First, you need to ensure that './data/' has the target file(xx.xlsx) you need, then execute _'featuresample.py'_ to extract 
 the corresponding molecular features, and then select the corresponding machine learning model training in the model file to 
 obtain the model and results.

1).Raw data analysis

   To ensure the reasonability of dividing, we used the Kullback-Leibler (KL) divergence to evaluate the similarity between the distribution of each subset events with varied event number and overall events. The more similar the two probability distributions are, the smaller the KL divergence is.
You can execute the **data_ana.py**, but it takes a long time.


2).Data feature extraction
    Make sure the following parameters are what you want at **featureSample.py**:
```python
...
file_name = [
            'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Cel-DPE-6SL-28930 events',
             'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Lac-DPE-6SL-27696 events',
             'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events'
             ]
...
num = 3  # divided into 3x3 domains
...
sample_len = 2000 # subsets, each of which contains 2000 events

```

3).Model training and testing:
    Make sure that the feature file is obtained from your previous data preprocessing, directly execute and select the ML model you need.

This example can be seen in **randomForest.py** 
```bash
cd train
python randomForest.py
```
So you can get a **_3ML_result.xlsx_** under the model folder, the **randomForest_model.m** is saved in the models folder.When you run the current script repeatedly, you need to save the previous model. Otherwise, it will be overwritten.

---------
## 3.Folder describing

1).Folder "data"

This Fold includes three subfolders:

- "3S3FL-MPB---6S3FL-MPB---6S2FL-MPB": Three MPB tagged sialylglycans containing branched fucose.

- "3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB": Four MPB tagged sialylglycans.

- "Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL": Three composite label tagged neutral disaccharides.

2).Folder "train"

This folder contains five machine learning models for training and testing.

3).Folder "models"

This folder is used to store model files for loading exiting models

4).Folder "picture"

Store some result pictures.

---------
## 4. code dependencies

```bash
pip install -r requirements.txt
```


