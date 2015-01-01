#!/usr/bin/python

import matplotlib.pyplot
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from time import time
import numpy as np
print "this is awesome"

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi","salary","total_payments","exercised_stock_options", 
"restricted_stock",  'total_stock_value',"to_poi_email","from_poi_email", "bonus",
"shared_receipt_with_poi",'expenses']

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
###NB Familiarizing with data
#print data_dict.keys()
#print data_dict.values()
print data_dict['LOCKHART EUGENE E']

for j in data_dict:
    key_count=0
    count=0
    for i in data_dict['SAVAGE FRANK'].keys():
        key_count+=1
        if data_dict[j][i]=="NaN":
            count+=1
    
    if count>19:
        print j, count
     
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)
def dict_to_list(key,normalizer):
    new_list=[]
    
    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append("NaN")     
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/data_dict[i][normalizer])
    return new_list
    
from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")
ctr=0
for i in data_dict:
    data_dict[i]["from_poi_email"]=from_poi_email[ctr]
    data_dict[i]["to_poi_email"]=to_poi_email[ctr]
    ctr+=1
#print data_dict['ALLEN PHILLIP K']
### we suggest removing any outliers before proceeding further
###NB observing data, removing outliers
features = [features_list[1], features_list[3],"poi"]

###NB removing total from data as well as a company name


## NB looking for additional outliers--didn't find any so whiting out print

data = featureFormat(data_dict, features)

for i in range(0,len(data)):
    xlab = data[i][0]
    ylab = data[i][1]
    if data[i][2]==0:
        matplotlib.pyplot.scatter( xlab, ylab,100,c='b')
        
    else:
        matplotlib.pyplot.scatter( xlab, ylab,100,c='r')


matplotlib.pyplot.xlabel(features[0])
matplotlib.pyplot.ylabel(features[1])
matplotlib.pyplot.show()

'''
for name in data_dict:
    if data_dict[name]["bonus"]>3500000 and data_dict[name]["bonus"]!="NaN":
        print name,data_dict[name]["salary"],data_dict[name]["bonus"]
    if data_dict[name]["salary"]>3500000 and data_dict[name]["salary"]!="NaN":
        print name,data_dict[name]["salary"],data_dict[name]["bonus"]
'''        
### Lay, Skilling, Pickering, Frevert all have high salaries but seem in line with level of position
### recommend not removing. Bonus outliers are bigwigs as well, do not remove
        
### if you are creating any new features, you might want to do that here
### store to my_dataset for easy export below
#new features proposed:  to_poi__email_scaled,from_poi_email_scaled,

   




def precision(pred,labels_test):
    precision, precision_d, precision_n=0.0,0.0,0.0
    for i in range(0,len(pred)):
        if pred[i]==1:
            precision_d+=1
            if labels_test[i]==1:
                precision_n+=1 
    if precision_d!=0:    
        precision=precision_n/precision_d
    return precision

def recall(pred,labels_test):
    recall, recall_d, recall_n=0.0,0.0,0.0
    for i in range(0,len(pred)):
        if labels_test[i]==1:
            recall_d+=1        
            if pred[i]==1:
                recall_n+=1
    if recall_d!=0:
        recall=recall_n/recall_d
    return recall

my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)



### if you are creating new features, could also do that here



### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)



### machine learning goes here!
### please name your classifier clf for easy export below

###NB inserted from cross-validation moduls
'''
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels, test_size=0.3, random_state=42)
'''

from sklearn.cross_validation import KFold
kf=KFold(len(labels),3, random_state=42)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
    

#scale data
min_max_scaler=preprocessing.MinMaxScaler()
features_train_scaled=min_max_scaler.fit_transform(features_train)

min_max_scaler=preprocessing.MinMaxScaler()
features_test_scaled=min_max_scaler.fit_transform(features_test)

print "Decision Tree Method"
from sklearn.tree import DecisionTreeClassifier
from sklearn import  grid_search

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'criterion': ('gini','entropy'),
              'splitter':('best','random'), 
              'min_samples_split':[4,5,10,20],
                'max_features':('auto','sqrt','log2',None),
                'max_depth':[None,1,2,10,50],
                'max_leaf_nodes':[None,8]}
clf = grid_search.GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

from sklearn.metrics import accuracy_score
acc=accuracy_score(labels_test, pred)
print acc
print precision(pred,labels_test)
print recall(pred,labels_test) 

#PCA analysis on features
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca=pca.fit(features_test)

features_train_pca=pca.transform(features_train)
features_test_pca=pca.transform(features_test)

print "SVM Method"
from sklearn.svm import SVC
clf=SVC()
clf.fit(features_train_pca,labels_train)
pred=clf.predict(features_test_pca)

acc=accuracy_score(labels_test, pred)
print acc
print precision(pred,labels_test)
print recall(pred,labels_test) 


#K means example to work on later
print "K Means Method"
from sklearn.cluster import KMeans
#param_grid = {'n_clusters':[8]}
#clf = grid_search.GridSearchCV(KMeans(random_state=42),param_grid)
clf=KMeans(n_clusters=3)
clf = clf.fit(features_train_pca)
pred=clf.predict(features_test_pca)
acc=accuracy_score(labels_test, pred)
#print("Best estimator found by grid search:")
#print(clf.best_estimator_)
print acc    
print precision(pred,labels_test)
print recall(pred,labels_test)   

print "Logistic Regression Method"
from  sklearn import linear_model
reg=linear_model.LogisticRegression()
reg=reg.fit(features_train,labels_train)
pred=reg.predict(features_test)
acc=accuracy_score(labels_test, pred)
print acc
print precision(pred,labels_test)
print recall(pred,labels_test) 
### Testing Precision, recall
#POIs in test set



### dump your classifier, dataset and features_list so 
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )



