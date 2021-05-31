# DisasterResponseProject

Author: Angelina Espinoza-Limon

This project presents the anaysis for the project Disaster Response. This analysis is performed over a set of messages that were captured with the Figure Eight system. The analysis concludes if a message belongs to some of the following categories: 'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'.

Each message might have several categories, then the problem requires a multi-class, multi-label and multi-output classifier. Therefore, the problem is addressed by coding a grid search to find out the best parameters for the ExtraTress and K-nearest classifiers, which are recomended for these kind of multi-class, multi-output classification.

Thereafter, it is firstly clean the data, then the messages tokens are extracted and its corresponding TF-IDF matrix is obtained. Then, it is defined a gridsearch and the hyperparameters options for the ExtraTress and K-nearest classifiers are set in the grid. This new model is then able to fit to predict the message categorie over test or unknown data. This analysis includes the bias-variance trade-off to evaluate the model precision, recall and the f1-score. 

The projects files are:

Data: 

  - disaster_categories.csv, disaster_messages.csv

Python files:

  - process_data.py
	This file contains the script for the Python code to load, clean and create the DisasterResponse database

  - train_classifier.py
     This file contains the script for the Python code to load from the database the messages (already cleaned), to train a model and to save the model into a pickle file (BestModelAndGridSearch.pkl) ready to be used.

Web App files:
















