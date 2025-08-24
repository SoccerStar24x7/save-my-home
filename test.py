import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import joblib

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)

label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

# converts "drug" column to numerical values using a map
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

# sets the input and target variables
X = my_data.drop(['Drug','Drug_num'], axis=1)
y = my_data['Drug']

# splits the data into training and testsets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=32) # test size is how fast the model learns

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4) # makes the decision tree
drugTree.fit(X_trainset,y_trainset) # trains the model!!



# the model now predicts
tree_predictions = drugTree.predict(X_testset)

print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))


# Assume 'model' is your trained DecisionTreeClassifier
joblib.dump(drugTree, 'decision_tree_model.joblib')

loaded_model = joblib.load('decision_tree_model.joblib')


