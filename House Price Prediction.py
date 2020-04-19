#!/usr/bin/env python
# coding: utf-8

# # House Price Predictor 
# ## Data ananysis using jupytor
# Here we are taking data from uci repository for housing price
# 
# 
# link : http://archive.ics.uci.edu/ml/machine-learning-databases/housing/
# 
# housing.data
# 
# 
# housing.names
# 
# copy these files in the working repository
# # making this data file to work for the project
# 
# convert the housing.data to .csv file using microsoft excel (should be properly delimitted for each column) 
# 
# * copy all the housing.data data to ms excel file 
# * select a col and then go to data tab and click on text-to-columns option and set the cols (14 cols)
# * create a blank row for setting heading for each col
# * you can write headings mannually also or can extract it from the housing.names by copying and pasting attribute(attribute information) part in excel and convert it using text-to-cols some how (heading of each col should be same as the heading given in the housing.names
# 
# * then finally save the file as .csv file in the same folder
# 
# 
# here i saved it as data.csv which 13 features and 1 label (14 cols). 
# 
# Now our raw data is ready to work with ml
# 
# ### Note: if data is given to you in .csv format then you don't need to format it you can use it directly for ml project
# 
# #### Note: before moving to the project you should know about each and every attributes in the data. how that attribute affect the problem negatively or positively. then you will be able to provide a solution for the given problem
# 
# 
# versions of library used here
# * python - 3.8.2
# * pip- 20.0.2
# * numpy-1.18.2
# * pandas-1.0.3
# * scikit-learn 0.22.2.post1 
# * matplotlib-3.2.1
# * sklearn 0.0
# * jupyter - 1.0.0
# * scipy-1.4.1
# 
# before going ahead please install required libraries using pip
# 
# Note :if you have different versions or higher versions then please check for functions which are not working in libraries and then update them by searching from google for new versions of library

# # data analysis using pandas
# pandas : pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language
# 

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv") # housing is a dataframe here openning using pandas


# In[3]:


housing.head()  # shows top 5 rows of the data set with heading


# In[4]:


housing.info() # informatoin about the data its attributes,datatypes,and much more


# here the no . of data is very short that is 506 because we are setting a demo for the analysis.
# 
# the process gonna be same for ml project with data set like in millions in real world.
# 
# taking small data set according to the machine we are having

# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe() #std : standard deviation 25% -25 percentile of data (50,75),min,max of cols


# In[7]:


# for plotting histogram
get_ipython().run_line_magic('matplotlib', 'inline')
# inline to see the plots in here using matplotlib
import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))  # to show histograms


# ## Now we will split the data for testing and training
# ### Train - Test Splitting
# 
#     by defining a function for splitting
#     it is only for learning purpose

# In[8]:


import numpy as np
def split_train_test(data, test_ratio):
    #randomly shuffling the data using numpy
    np.random.seed(42) # for fixing the random suffling so that it will be fixed shuffling everytime
    shuffled = np.random.permutation(len(data)) 
    #print(shuffled)
    #now split
    #generally test ratio is 80%-20%
    test_set_size = int(len(data) * test_ratio)  #506 * 0.2   or 20%
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[9]:


# train_set, test_set = split_train_test(housing, 0.2)


# In[10]:


# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# ## Train - test splitting using sklearn 
# we don't need to define a function it is already in sklearn package

# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# there is problem here : there may be condition that in training data set we don't cover all kind of population(type of data) present in dataset e.g. in CHAS there are online 35 values which has 1 and 471 has 0 so there may be condition that all 1's will go to test_set and our model will be trained with only 0 value of CHAS which can lead to error in the model because this feature may be very important to know . so while training we will have to cover all kind population given in the data. so we will use stratified sampling to sample the training and testing data.
# 
# 
# stratified sampling 
# 
# ### Stratified sampling
# HERE WE WILL BE DOING FOR COL 'CHAS' (WHAT EVER COL IS IMPORTANT USE THAT HERE) SPECIFIED BY THE COMPANY

# In[12]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
#splitting the data by the CHAS values in both the datasets
# n-splits : no. of re-shuffling


# In[13]:


# strat_test_set
# strat_test_set.describe()
strat_test_set.info()
strat_test_set['CHAS'].value_counts()



# In[14]:


strat_train_set['CHAS'].value_counts()


# In[15]:


print(95/7)
print(376/28) #you will see here no. of 0's and 1's are equally distributted in both the datasets


# In[16]:


# from here housing will be the training data set
#setting copy of training data set to housing after shuffling
housing = strat_train_set.copy()  # now housing will contain training data only


# ## looking for correlation
# 
# we are talking about pearson correlation 
# 
# correlation = how strongly a variable depends on the output
#    *  1- strong positive correlation
#    *  0- no correlation
#    *  -1- weak correlation / strong negative correlation
#    
# correlation lie between -1 and 1

# In[17]:


#creating correlation matrix
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


#plotting correlations for some attributes
from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize = (15,8))


# In[19]:


housing.plot(kind="scatter", x="RM", y="MEDV",alpha=0.8)   # RM vs MEDV  +ve correlation


# ## Trying out attribute combinations
# 

# In[20]:


housing['TAXRM']= housing['TAX']/housing['RM'] #creating a new attribute by combining TAX and RM and try to see the impact
#TAX per number of rooms in the house


# In[21]:


housing.head() #now you will 15 cols in data
# NOte : it will not affect the original data.csv file it is only for the housing dataframe in this program


# In[22]:


#creating correlation matrix
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV",alpha=0.8)  #plotting TAXRM with MEDV
# you will see here a -ve correlation


# ## setting train data set featues and labels

# In[24]:


# MEDV is the label col 
# we are not adding here the new extra col "TAXRM" to train the model
# it was just for not standard data 
# here we are working on a standard data set from uci repository
housing = strat_train_set.drop("MEDV", axis=1) #features 1-13 col
housing_labels = strat_train_set["MEDV"].copy() #labels last col


# ## Missing Attributes
# Means if some of the data is missing from a col/attribute
# 
# to take care of missing attributes,you have three options:
#          1. Get rid of the missing data points
#          2. Get rid of the whole attribute
#          3. Set the vlaue to some value(0, mean, median)

# In[25]:


#let assume that some that is missing in attribute RM
a= housing.dropna(subset=["RM"])  #option1
a.shape
# Note that the original housing dataframe will remain unchanged


# In[26]:


housing.drop("RM", axis=1).shape #option2
#Note that there is no RM column and also note that the original housing dataframe will ramain unchanged


# In[27]:


median = housing["RM"].median() #compute median for option3


# In[28]:


housing["RM"].fillna(median) #option3
#note that the original housing dataframe will remain unchanged


# In[29]:


housing.shape


# In[30]:


housing.describe()  #before we started imputer

# RM has 3 missing values in the dataset


# ## Now we will be doing this using sklearn

# In[31]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[32]:


imputer.statistics_.shape


# In[33]:


imputer.statistics_   #median of each cols


# ### Now we will  fit value in any missing values in any column in the data

# In[34]:


X = imputer.transform(housing)


# In[35]:


housing_tr = pd.DataFrame(X, columns=housing.columns) # create new dataframe with housing for training data set 


# In[36]:


housing_tr.describe()  #after applying imputer in housing and storing it to housing_tr
#now no missing values are there


# #### upto this step we were analysing the data , understanding its patthern, and understanding the data , Now we prepare  the model

# ## Scikit-learn Design

# Primarily, there are three types of objects
# 1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has fit() method and transform() method. Fit method - Fits the dataset and calculates internal parameters
# 
# 2. Transformers - transform method takes input and returns output based on the learnings from fit() . It also has a convenience function called fit_transform() which fits and then transforms.
# 
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.

# ## Feature Scaling
# making all feature values to a similar kind of scaling

# Primarily, two types of feature scaling methods:
# 1. Min-max scaling (Normalization)
#     * (value-min)/(max-min)
#     * Sklearn provides a class called MinMaxScaler for this
# 2. Standardization
#     * (value - mean)/std
#     * Sklearn provides a class called standard scaler for this
#     
# std - >standard deviation
# 
# we will use standardization here

# # Creating a Pipeline
# pipeline : so that we can make changes to our model easily later, doing series of steps in the problem, so that we can automate the task
# 
# using sklearn: it provide piplining also

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
#     .... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[38]:


housing_num_tr= my_pipeline.fit_transform(housing)


# In[39]:


housing_num_tr #it is an numpy array


# In[40]:


housing_num_tr.shape  # for train dataset


# # Selecting a desired model for this House Price Prediction problem

# ##  You can change here different models for results don't need to change anywhere

# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#using Linear
# model = LinearRegression() 
#usng tree regressor
# model = DecisionTreeRegressor() 
#using random forest
model = RandomForestRegressor()

model.fit(housing_num_tr, housing_labels)  #giving features and lables of training data to fit the model


# ## now model has been fitted

# In[42]:


#check for some data from training dataset
some_data = housing.iloc[:5]


# In[43]:


some_labels = housing_labels.iloc[:5]


# In[44]:


prepared_data = my_pipeline.transform(some_data)


# In[45]:


model.predict(prepared_data)  #predicted output numpy array


# In[46]:


list(some_labels)  #original output


# ## Evaluating the model
# using Root mean square error

# In[47]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions) #parameter actual labels, predicted labels
rmse = np.sqrt(mse)


# In[48]:


rmse


# it shows overfitting for tree regressor

# ## Using better evaluation technique - cross validation

# making 10 groups of training data  1 2 3 4 5 6 7 8 9 10
# 
# then take 1 group for testing and 9 group for training and find error
# 
# do this for all the ten groups and then find the combined error

# In[49]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
#cv = 10 means go for 10 folds
rmse_scores = np.sqrt(-scores)


# In[50]:


rmse_scores #numpy array


# using this evaluation for tree regression it looks like this is better than linear regression

# In[51]:


def print_scores(scores):
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("Standard deviation" , scores.std())


# In[52]:


print_scores(rmse_scores)


# #### make python file for this model to run on visual studio 
# #### now create a sklearn joblib for this 

# # Saving the model

# In[53]:


from joblib import dump, load
dump(model, 'HPP.joblib')


# # Testing the model on test data

# In[54]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions) 
final_rmse = np.sqrt(final_mse)


# In[55]:


final_rmse


# In[56]:


print(final_predictions, list(Y_test))


# In[57]:


prepared_data[0]


# In[58]:


some_labels


# # Using the model to take output

# In[59]:


from joblib import dump, load
import numpy as np
model = load('HPP.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.841041, -1.312238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.456164221, -0.86091034]])
model.predict(features)


# In[ ]:




