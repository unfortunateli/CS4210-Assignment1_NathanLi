#-------------------------------------------------------------------------
# AUTHOR: Nathan Li
# FILENAME: knn.py
# SPECIFICATION: a program to classify emails using the K-Nearest Neighbors algorithm based on the data provided in the file 'email_classification.csv'
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

#Loop your data to allow each instance to be your test set
errorCount = 0
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    for j in db:
        if j != i:
            X.append([float(feature) for feature in j[:-1]])

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for j in db:
        if j != i:
            if j[-1] == 'spam':
                Y.append(1)
            elif j[-1] == 'ham':
                Y.append(0)
            

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(feature) for feature in i[:-1]]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    true_label = 1 if i[-1] == 'spam' else 0

    if class_predicted != true_label:
        errorCount += 1

#Print the error rate
#--> add your Python code here
errorRate = errorCount / len(db)
print("Error rate:", errorRate)







