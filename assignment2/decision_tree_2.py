#-------------------------------------------------------------------------
# AUTHOR: Nathan li
# FILENAME: decision_tree_2.py
# SPECIFICATION: a program to predict whether a patient should be fitted with a contact lens or not based on the data provided in the file 'contact_lens_training_1.csv' and 'contact_lens_test.csv'
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    dbTraining = []
    df_train = pd.read_csv(ds)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for _, row in df_train.iterrows():
        dbTraining.append(row.tolist())

    for _, row in df_train.iterrows():
        features = []
        for i in range(4):
            if row.iloc[i] == 'Young':
                features.append(1)
            elif row.iloc[i] == 'Prepresbyopic':
                features.append(2)
            elif row.iloc[i] == 'Presbyopic':
                features.append(3)
            elif row.iloc[i] == 'Myope':
                features.append(1)
            elif row.iloc[i] == 'Hypermetrope':
                features.append(2)
            elif row.iloc[i] == 'No':
                features.append(1)
            elif row.iloc[i] == 'Yes':
                features.append(2)
            elif row.iloc[i] == 'Reduced':
                features.append(1)
            elif row.iloc[i] == 'Normal':
                features.append(2)                
        X.append(features)

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for _, row in df_train.iterrows():
        if row.iloc[4] == 'Yes':
            Y.append(1)
        else:
            Y.append(2)

    total_accuracy = 0

    #Loop your training and test tasks 10 times here
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> addd your Python code here
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)
       
       correct_predictions = 0
       incorrect_predictions = 0

       #Read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = []
       df_test = pd.read_csv('contact_lens_test.csv')
       for _, row in df_test.iterrows():
           dbTest.append(row.tolist())

       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            features = []
            for j in range(4):
                if data[j] == 'Young':
                    features.append(1)
                elif data[j] == 'Prepresbyopic':
                    features.append(2)
                elif data[j] == 'Presbyopic':
                    features.append(3)
                elif data[j] == 'Myope':
                    features.append(1)
                elif data[j] == 'Hypermetrope':
                    features.append(2)
                elif data[j] == 'No':
                    features.append(1)
                elif data[j] == 'Yes':
                    features.append(2)
                elif data[j] == 'Reduced':
                    features.append(1)
                elif data[j] == 'Normal':
                    features.append(2)                    

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            if data[4] == 'Yes':
                true_label = 1
            else:
                true_label = 2

            class_predicted = clf.predict([features])[0]

            if class_predicted == true_label:
                # increment the count of correct predictions
                correct_predictions += 1
            else:
                # increment the count of incorrect predictions
                incorrect_predictions += 1

       run_accuracy = correct_predictions / len(dbTest)
       total_accuracy += run_accuracy

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    average_accuracy = total_accuracy / 10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f'final accuracy when training on {ds}: {average_accuracy:.2f}')
