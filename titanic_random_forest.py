# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:31:47 2017

@author: Andre
"""

#%%
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import f1_score

#upload data
data_train = pd.read_csv("train.csv")
label_train = data_train["Survived"]
data_test = pd.read_csv("test.csv")

#merge data
merge = data_train.append(data_test, ignore_index = True)

merge.shape

#%%
merge["Age"] = merge["Age"].fillna(merge["Age"].mean())
merge["Fare"] = merge["Fare"].fillna(merge["Fare"].mean())

merge.loc[merge["Sex"] == "male", "Sex"] = 0
merge.loc[merge["Sex"] == "female", "Sex"] = 1

embarked = pd.get_dummies(merge["Embarked"], prefix = "Embarked")

pclass = pd.get_dummies(merge["Pclass"], prefix = "Pclass")

merge["Cabin"] = merge["Cabin"].fillna("U")
merge["Cabin"] = merge["Cabin"].map( lambda c : c[0] )
cabin = pd.get_dummies(merge['Cabin'], prefix = 'Cabin')

#%%

title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = merge[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# a map of more aggregated titles
Title_Dictionary = {
                   "Capt":       "Officer",
                   "Col":        "Officer",
                   "Major":      "Officer",
                   "Jonkheer":   "Royalty",
                   "Don":        "Royalty",
                   "Sir" :       "Royalty",
                   "Dr":         "Officer",
                   "Rev":        "Officer",
                   "the Countess":"Royalty",
                   "Dona":       "Royalty",
                   "Mme":        "Mrs",
                   "Mlle":       "Miss",
                   "Ms":         "Mrs",
                   "Mr" :        "Mr",
                   "Mrs" :       "Mrs",
                   "Miss" :      "Miss",
                   "Master" :    "Master",
                   "Lady" :      "Royalty"

                   }

# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )
#title = pd.concat( [ title , titles_dummies ] , axis = 1 )

title.head()

#%%
# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
   ticket = ticket.replace( '.' , '' )
   ticket = ticket.replace( '/' , '' )
   ticket = ticket.split()
   ticket = map( lambda t : t.strip() , ticket )
   ticket = list(filter( lambda t : not t.isdigit() , ticket ))
   if len( ticket ) > 0:
       return ticket[0]
   else:
       return 'XXX'

ticket = pd.DataFrame()

# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = merge[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

ticket.shape
ticket.head()

#%%

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = merge[ 'Parch' ] + merge[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

family.head()

#%%

data = pd.concat([merge["PassengerId"], merge["Age"], merge["Fare"], merge["Sex"], merge["Survived"], embarked, pclass, cabin, title, ticket, family], axis = 1)
data.head()

# %%

real_data = data[:891]
real_label = real_data["Survived"]
real_data = real_data.drop("Survived", axis = 1)

test_data = data[891:]
test_data = test_data.drop("Survived", axis = 1)

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(real_data, real_label, test_size=0.25, random_state=0)

# %% Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def fit_model(X, y):
   from sklearn.model_selection import GridSearchCV
   from sklearn.metrics import make_scorer

   parameters = {'min_samples_split' : [2, 5, 10, 25, 50],
                 'n_estimators' : [1, 5, 10, 25, 50],
                   'max_depth' : [1, 2, 3, 4, 5],
                   'min_samples_leaf' : [1, 2, 3, 4, 5]}
   
   # TODO: Initialize the classifier
   clf = RandomForestClassifier(random_state = 0)
   
   # TODO: Make an f1 scoring function using 'make_scorer'
   #f1 = f1_score(X_test, y_test)
   f1_scorer = make_scorer(f1_score, pos_label = 1)
   
   # TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
   grid_obj = GridSearchCV(clf, param_grid=parameters, scoring= f1_scorer)
   
   # TODO: Fit the grid search object to the training data and find the optimal parameters
   grid_obj = grid_obj.fit(X, y)
   
   # Get the estimator
   clf = grid_obj.best_estimator_
   print clf
   return clf
   
# Report the final F1 score for training and testing after parameter tuning
#print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
#print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
some = []
reg = fit_model(xtrain, ytrain)
for i, Survived in enumerate(reg.predict(test_data)):
   print i+1,Survived
   some.append(Survived)
# %%
submission = pd.DataFrame({
       "PassengerId": test_data["PassengerId"],
       "Survived": some
   })
print submission
submission.to_csv("kaggle2.csv", index=False)

#%%


actual = 0
new = 0
index_i = 0
index_j = 0
for i in range(1, 20):
   for j in range(2, 301):
       print i
       print j
       clf_A = RandomForestClassifier(n_estimators=i, min_samples_split = j, random_state=0)
       new = train_predict(clf_A, xtrain, ytrain, xtest, ytest)
       if actual < new:
           index_i = i
           index_j = j
           actual = new
           
print actual
print index_i
print index_j

#%%

goal = train_predict(reg, xtrain, ytrain, xtest, ytest)
print goal