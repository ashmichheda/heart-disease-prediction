# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from os import listdir
from sklearn.utils import shuffle
import sklearn.utils
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


data2 = pd.DataFrame()    

filenames = find_csv_filenames(r"./CSV")
for name in filenames:
    data = pd.read_csv(r"./CSV/"+name) 
    data.to_csv(r"./CSV/"+name)

filenames = find_csv_filenames(r"./CSV")
for name in filenames:
    data1 = pd.read_csv(r"./CSV/"+name) 
    data2=data2.append(data1)

#shuffle data
data2 = shuffle(data2)
data2 = data2.sample(frac=1).reset_index(drop=True)
data2 = sklearn.utils.shuffle(data2)
data2 = data2.reset_index(drop=True)

X = data2

y = pd.DataFrame(data=data2, columns=['num'])

X.drop(X.iloc[:, 0:1], inplace=True, axis=1)

del X['num']

sign = {0 : "No chance", 1: "High Chance"}
#sign = {0 : "No chance", 1: "Low Chance",2 : "Moderate chance", 3: "High chance",4 : "Critical"}
#sign = {"No chance": 0, "Low Chance": 1,"Moderate chance" : 2, "High Chance" : 3 ,"Critical": 4}
y.num = [sign[item] for item in y.num]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 20) 
#print(OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test))
model=KNeighborsClassifier(n_neighbors = 7, metric = "minkowski")
model.fit(X_train, y_train)
pred=model.predict(X_test)
print("Acc:",model.score(X_test,y_test))
accuracy = accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# # creating a confusion matrix 
cm = confusion_matrix(y_test, pred) 

print("Confusion matrix: \n",cm)

print("Classification Report: ")
print(classification_report(y_test, pred))
