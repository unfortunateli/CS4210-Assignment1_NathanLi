#-------------------------------------------------------------------------
# AUTHOR: Nathan Li
# FILENAME: decision_tree.py
# SPECIFICATION: Derive a depth-2 decision tree to classify the contact lens data set
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here
dimensions = len(db[0]) - 1    # number of features/classes
featureValues = []
for col in range(dimensions): 
    values = sorted(set(row[col].strip() for row in db))    #sort alphabetically
    featureValues.append(values)

X = []
for row in db:
    newRow = []
    for col in range(dimensions):
        newRow.append(featureValues[col].index(row[col].strip()))
    X.append(newRow)

#encode the original categorical training classes into numbers and add to the vector Y.
#--> addd your Python code here
classValues = sorted(set(row[dimensions].strip() for row in db))    #sort alphabetically
map = {}
for i, value in enumerate(classValues):
    map[value] = i

Y = []
for row in db:
    Y.append(map[row[dimensions].strip()])


#fitting the depth-2 decision tree to the data using entropy as your impurity measure
#--> addd your Python code here
#clf =
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=0)
clf = clf.fit(X, Y)

#plotting decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()