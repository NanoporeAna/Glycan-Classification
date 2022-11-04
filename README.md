# Glycan-Classification
1.Software description                               
                                                                                
 RNA-Classification serves as a machine learning-based nanopore analysis platform 
 for the identification of different RNA types.                 
 The software contains three functions as follows:                                
                                                                               
 1).Model training by the training dataset.	            
                                                                                
 2).Model testing by the testing dataset and output model result.                                   
                                                                                
 3).Model predicting by the new dataset                                          


 2.Operating procedures                             

Run RNA-Classification.exe

 Here, Whether to load the model is need to choose
 -Load the previous saved model?[y]es/[n]o 

If "No" is selected, the software proceeds to model training and testing:
1).Model training and testing

   Model training:
	The following instruction for model training:

	Start training the models...

	The times of iteration is need to input to begin the model training
	-Please input the times of iteration: example:10
        		The time of model building is about: 30 minutes
	
	After the model training, you can choose whether to save the trained model
	-Save the training model?[y]es/[n]o example:y

	If "Yes" is selected, the trained model will be saved to local:
	Model save finish...

   Model testing:
	The following three are the output results of the model in turn, that are feature importance, 
	confusion matrix and learning curve:

 	-Output feature_importances?[y]es/[n]o example:y
 	-Output confusion_matrix？[y]es/[n]o example:y
 	-Output learning_curve？[y]es/[n]o example:y
         		The time of output learning_curve is about: 4 minutes 40s

If "Yes" is selected, the software proceeds to model predicting:
2). Model predicting

After the model is trained, you can choose whether to predict the new sample
 -Predicting data？[y]es/[n]o example:y

If "Yes" is selected, the original file of the predicted sample (.npz format) need to be entered:
 -The raw file to be tested(*.npz): example:.\dataset\pre_set\overhanged_sirna\overhanged_sirna.npz
          The time of loading the demo dataset is about: 12s

Select whether to output feature lists and event images:
 -Output the feature table? [y]es/[n]o  example: y
 -Output the picture?[y]es/[n]o example:y
          The time of output the predict demo dataset is about: 1 minutes 56s
This procedure can predict multiple samples continuously:
 -Predict other data？[y]es/[n]o example:n

3.Folder describing

1).Folder "dataset"

This Fold includes three subfolders:"train_set","test_set","pre_set"
 --"train_set": contains raw data and feature data from the training dataset and the label of training dataset
 --"test_set":contains raw data and feature data from the test dataset and the label of test dataset
 --"pre_set":The original event file that contains mixed samples

2).Folder "model"

This folder is mainly used to store model files for loading existing models

3).Folder "output"

All output files are stored here. Each folder is named with the time of the output file.The folder contains 3 subfolders:
"classification_data","feature_data" and "model_data"
 --"classification_data": contains two subfolders "data" and "picture"
          --"data": contains the predict output file-output_predict.txt
          --"picture": the folder containing all RNA classes, each containing a picture of the predicted event for the 
                            corresponding RNA class
 --"feature_data": contains one fold "picture" and two file "X.npz", "X.txt"
          --"X.npz": the output feature file of the predicting raw data
          --"X.txt":the output feature table of the predicting feature data
          --"picture":the output event image from the the predicting raw data
 --"model_data": contains four file "confusion_matrix.png", "features_importances.png", 
    "learning.txt" and "learning_curve_classifar.png"
          --"confusion_matrix.png":If you select the output confusion matrix, it will be output here
          --"features_importances.png":If you select the output features_importances, it will be output here
          --"learning.txt":If you select the output learning_curve, it will be output here
          --"learning_curve_classifar.png":If you select the output learning_curve, it will be output here

4). soft dependencies 
For the original code, the software uses the following main dependencies:
scipy
past
sklearn
tqdm
os
xgboost
matplotlib
time
collections
neo
pylab
pandas
decimal
