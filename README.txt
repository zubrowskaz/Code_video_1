Last updated: 12/6/2019

Questions:
Email chris@cs.pitt.edu

Code requires PyTorch and Python 3.7+. Anaconda distribution is highly recommended.

This code implements our model as presented in our NeurIPS 2019 paper.

It is assumed that the dataset is in the same folder as this code. Alternatively, create a symbolic link to the imgs folder of the dataset in the directory of the code.

First, run train_model.py to train a stage 1 model. We provide the precomputed Doc2Vec features as an attached pickle file. If you wish to run on your own data, first run GenSym on your own documents and extract Doc2Vec features. 

Next, run extract_features.py after choosing the best model on the validation set. We also provide pre-extracted features for the test set.

Finally, run train_eval_stage_2.py which trains the classifier using the extracted features and evaluates it on the test set.

We also provide a pretrained classifier model. Note that due to random variations in training and initialization, results may slightly deviate from the paper each time the code is run. If you wish to train your own model, you must first extract features on the train / test dataset and uncomment lines to train in the stage 2 file.