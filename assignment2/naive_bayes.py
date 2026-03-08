#-------------------------------------------------------------------------
# AUTHOR: Nathan Li
# FILENAME: naive_bayes.py
# SPECIFICATION: a program to classify weather instances using the Naive Bayes algorithm
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for row in dbTraining:
    outlook  = row[1]
    temperature = row[2]
    humidity = row[3]
    wind = row[4]

    if outlook == 'Sunny':
        outlook = 1
    elif outlook == 'Overcast':
        outlook = 2
    elif outlook == 'Rain':
        outlook = 3

    if temperature == 'Hot':
        temperature = 1
    elif temperature == 'Mild':
        temperature = 2
    elif temperature == 'Cool':
        temperature = 3

    if humidity == 'High':
        humidity = 1
    elif humidity == 'Normal':
        humidity = 2

    if wind == 'Strong':
        wind = 1
    elif wind == 'Weak':
        wind = 2

    X.append([outlook, temperature, humidity, wind])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for row in dbTraining:
    if row[5] == 'Yes':
        Y.append(1)
    else:
        Y.append(2)

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf = clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print('{:<6} {:<10} {:<12} {:<8} {:<8} {:<11} {}'.format('Day', 'Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis', 'Confidence'))

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
    day = row[0]
    outlook  = row[1]
    temperature = row[2]
    humidity = row[3]
    wind = row[4]

    if outlook == 'Sunny':
        outlook = 1
    elif outlook == 'Overcast':
        outlook = 2
    elif outlook == 'Rain':
        outlook = 3

    if temperature == 'Hot':
        temperature = 1
    elif temperature == 'Mild':
        temperature = 2
    elif temperature == 'Cool':
        temperature = 3

    if humidity == 'High':
        humidity = 1
    elif humidity == 'Normal':
        humidity = 2

    if wind == 'Strong':
        wind = 1
    elif wind == 'Weak':
        wind = 2

    testSample = [outlook, temperature, humidity, wind]
    probs = clf.predict_proba([testSample])[0]
    prediction = clf.predict([testSample])[0]
    confidence = max(probs)

    #Compare the prediction with the true label (located at row[5]) of the test instance to calculate the confidence of the prediction.
    #--> add your Python code here
    if prediction == 1:
        label = 'Yes'
    else:
        label = 'No'
    #Print the test sample with its predicted class and the confidence of the prediction. For instance:
    #print("Sunny", "Hot", "High", "Weak", "No", "0.8")
    #--> add your Python code here
    if confidence >= 0.75:
        print('{:<6} {:<10} {:<12} {:<8} {:<8} {:<11} {:.2f}'.format(row[0], row[1], row[2], row[3], row[4], label, confidence))


