# 100 DAYS OF MACHINE LEARNING
___

This repo consists of my whole 100 days of learning journey and in this file I will be documenting this complete journey !! Let's go !!
___
## Syllabus to cover
This is just a pre-setup and things are added as exploration continues !!

| **S.N.** | **Books and Lessons (Resources)**                                                                                                 | **Status** |
|----------|-----------------------------------------------------------------------------------------------------------------------------------|------------| 
| **1.**   | [**Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow**](https://github.com/ageron/handson-ml3)                   | ⏳          |
| **2.**   | [**Machine Learning Scientist With Python**](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python) | ⏳          |

___

## Projects

| **S.N.** | **Project Title**                                                                                                                                                                                | **Status** |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| 1.       | [**California Housing Price Prediction**](https://github.com/iamshishirbhattarai/Machine-Learning/blob/main/California%20Housing%20Price%20Prediction/California_housing_price_prediction.ipynb) | ⏳          |
## Topics Learnt Every Day

| **Days**         | **Learnt Topics**                                              | **Resources used**                                                                                          |
|------------------|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [Day 1](Day1) | EDA, Splitting with random & stratified sampling, correlations | [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://github.com/ageron/handson-ml3) |
 | [Day 2](Day2) | Imputation, Estimators, Transformers, Predictors, get_dummies vs OneHotEncoder | [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://github.com/ageron/handson-ml3)|


___

## Day 1 

### California Housing Price Prediction
Today I started to actually create my notebook on 'California housing price prediction' : a project from the book 'Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow'.
The tasks I performed and learnt are listed below:

- Loaded the dataset.
<br> <br>
- Observed dataset distribution. <br> <br>
  ![dataset_distribution](Day1/dataset_distribution.png)


- Split the dataset into training and testing set using both 
 random and stratified sampling.  <br>
    <br>
    1. **Random Sampling :** 
            It is a technique of selecting subset of datas from large dataset in such
  a way that there is an equal chance of each points to be selected. This method doesn't
  introduce any kind of biases.
<br> <br>
  2. **Stratified Sampling :** 
            It is a technique that is used when we have to deal with imbalanced
  datasets where some datasets are under-presented. It works by providing the
  equal proportion of target variable 'y' of each class in both training and testing sets.
  <br> <br>
- Observed the geographical_distribution. Here, the
  size of the points determines the size of *population* and color represents
  the *median_house_value*. <br> <br>
 ![detailed_geographical_observation](Day1/detailed_geographical_observation.png)
- Studied correlations among different variables and found out that the 
*median_income* has good correlation with *median_house_value*. <br>  
     ![corr_income_value](Day1/corr_income_value.png)

___

## Day 2

### Continuing California Housing Price Prediction

I continued the ongoing project i.e. **California Housing Price Prediction** and got to apply some exciting stuffs that I
have learnt. Key learnings and tasks performed are listed below:
<br> <br>
- Experimented with various attributes combinations.
``` python
    #Finding room per house
    housing['room_per_house'] = housing['total_rooms'] / housing['households']
    #Finding bedrooms_ratio 
    housing['bedrooms_ratio'] = housing['total_bedrooms'] / housing['total_rooms']
    #Finding people_per_house
    housing['people_per_house'] = housing['population'] / housing['households'] 
```
- Performed **Imputation** on missing datas using **median** strategy and converted transformed dataset to a DataFrame
   <br> <br> **Imputation**: It is the process of setting null values to some values such as zero, the mean, the median,
etc. I used Scikit-learn's class **SimpleImputer** and applied **median** strategy. The code snippet is as follows:
``` python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")

#Separating numerical-valued attributes as housing_num
housing_num = housing.select_dtypes(include = [np.number])
imputer.fit(housing_num)
X = imputer.transform(housing_num)
#Converting X into a dataframe
housing_tr = pd.DataFrame(X, columns = housing_num.columns,
                          index = housing_num.index)
```
- Learnt about some frequently used terminologies **estimators**, **transformers** and **predictors** in Scikit Learn.
    <br> <br> 
        **i. Estimators :** Any object that estimates some parameters on the basis of dataset is an estimator. It is
    performed by *fit()* function. **SimpleImputer** is an example of estimator. <br> <br>
        **ii. Transformers :** Any object that is capable to transform the dataset is a transformer. It uses 
    *tranform()* function. **SimpleImputer** is also a transformer. Both estimation & transform can be done at once by
    using *fit_transform()* method. <br> <br>
        **iii. Predictors :** The estimator capable of making predictions with a dataset given is a predictor. 
    **LinearRegression* is a predictor which uses **predict()** function to make predictions. <br> <br>
- Understood the differences in using *get_dummies* from *pandas* and *OneHotEncoder* from *Scikit-Learn*
   <br> <br> ![get_dummies_VS_OneHotEncoder](Day2/get_dummies_VS_OneHotEncoder.png)
        <br> 
   *OneHotEncoder* remembers which categories it was trained on while *get_dummies* doesn't remember. As shown in above
screenshot of notebook where both were trained on same datas. In case of pandas, when dataset with unknown categories
was asked to transform, it happily regenerated column for even unknown categories while *OneHotEncoder* throws error !!

___