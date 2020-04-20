#required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#loading data
housing = pd.read_csv("data.csv")


#Test - Test splitting using sklearn
#splitting in 80% and 20% ratio
#-------------------------------------------------------------
#1. using simple splitter
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)


#OR
#2. using stratified shuffling for col 'CHAS'
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    train_set = housing.loc[train_index]
    test_set = housing.loc[test_index]
# print(train_set)
# print(test_set)
#--------------------------------------------------------------

#training  set in housing annd housing_labels
housing = train_set.drop("MEDV", axis=1) #features 1-13 col
housing_labels = train_set["MEDV"].copy() #labels last col

#creating pipline using sklearn for the model
my_pipeline = Pipeline([
    # imputer for : fillig missing data 
    ('imputer', SimpleImputer(strategy="median")),
    #.... add as many as you want in your pipeline
    # standard scaler : for scaling variables of the data to a similar kind
    ('std_scaler', StandardScaler()),
])

#passing the training features through the pipeline to fit it
#it returns a numpy array
housing_num_tr= my_pipeline.fit_transform(housing)

#Now training the model with different type of model
#---------------------------------------------------------------------------
#1. using Linear
# model = LinearRegression() 
#2. usng tree regressor : it is better than linear regressor
# model = DecisionTreeRegressor() 
#3. using random forest  : it is better than above two as we see in the jupyter notebook
model = RandomForestRegressor()
#----------------------------------------------------------------------------
#final training
#passing features and labels
model.fit(housing_num_tr, housing_labels)


# Testing the model with test_data
X_test = test_set.drop("MEDV", axis=1)
Y_test = test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions) 
final_rmse = np.sqrt(final_mse)

print("\nRoot Mean squared error with test_data",final_rmse)

#final predictions
# print(final_predictions, list(Y_test))

#input new features
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])

#output the predicted amount by the model 
print("Predicted Amount by the model :",model.predict(features))

