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
 | [Day 3](Day3) | Feature Scaling, Custom Transformers, Pipelines, ColumnTransformers | [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://github.com/ageron/handson-ml3) |



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

## Day 3
### Continued California Housing Price Prediction

Today I learnt a very crucial step in Machine Learning i.e. **Feature Scaling And Transformation**. Below are few
summarization of my learnings and performed tasks:

- Performed two types of scaling: <br> <br>**i. Min-Max Scaling :** For this scaling I used *Scikit-Learn* 's *MinMaxScaler* class. This is the simplest
scaling method that is performed by subtracting the min value and dividing by the difference of the min and the max.

    ```python
    from sklearn.preprocessing import MinMaxScaler
    
    min_max_scaler = MinMaxScaler(feature_range = (-1,1))
    housing_min_max_scaled = min_max_scaler.fit_transform(housing_num) 
   ```

    **ii. Standarization :** For this scaling I used *Scikit-Learn* 's *StandardScaler* class. It is performed by subtracting
with mean and dividing by the standard deviation.

    ```python
    from sklearn.preprocessing import StandardScaler
    
    std_scaler = StandardScaler()
    housing_num_std_scaled = std_scaler.fit_transform(housing_num)
    ```


- Learnt about transforming target values. When we transform target values, it is very necessary to inverse the 
transformation while providing the predicted values. Suppose we replaced the target value with it's logarithm, the
output will also be in logarithm. So, for this I can use *inverse_transform()* method from Scikit's Learn. But, rather
I chose a simpler option and decided to use *TransformedTargetRegressor* that simply provides the output by inversing
the transform. 

    ```python
    #Using TransformedTargetRegressor
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import LinearRegression
    
    model = TransformedTargetRegressor(LinearRegression(), transformer = StandardScaler())
    model.fit(housing[["median_income"]], housing_labels)
    some_new_data = housing[["median_income"]].iloc[:5]
    predictions = model.predict(some_new_data)
    ```
- Learnt to make simple **custom transformer**. Here, I simply transformed the *population* as logarithm of *population*
and visualize the original and logarithmic population to see the changes in distribution of the data.    
    ```python
    #Applying logarithmic transformation in population as the datas are skewed
    from sklearn.preprocessing import FunctionTransformer
    
    log_transformer = FunctionTransformer(np.log, inverse_func = np.exp)
    log_pop = log_transformer.transform(housing['population'])
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))
    ax[0].hist(housing['population'], bins=50)
    ax[0].set_xlabel('Original Populaton')
    ax[1].hist(log_pop, bins=50)
    ax[1].set_xlabel('Log of Population')
    plt.savefig("log_population_vs_population.png", dpi = 300)
    plt.show()
    ```
  **Output :** <br><br>
  ![log_population_vs_population](Day3/log_population_vs_population.png)  
<br> <br>
- Applied **pipelines** and **ColumnTransformers** 
<br> <br>
**i. Pipelines :** It is simply a sequence of transformations. For this, we can use *Pipeline* or *make_pipeline* class
from *Scikit-Learn*. If we prefer *Pipeline* we need to provide name/estimators pairs defining a sequence of steps, while
in *make_pipeline* we can simply provide estimators. <br> <br>
**Pipeline Implementation Code :**
    ```python
    from sklearn.pipeline import Pipeline
    num_pipeline = Pipeline([
       ("impute", SimpleImputer(strategy="median")),
       ("standardize", StandardScaler()),
    ])
   ```

    **make_pipeline Implementation Code:**
    ```python
    from sklearn.pipeline import make_pipeline
    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    ```
  
    **ii. ColumnTransformers :** The *ColumnTransformer* in *scikit-learn* is a powerful tool that allows you to apply 
different preprocessing pipelines to different subsets of features within your dataset. This is particularly useful
when you have a mix of numerical, categorical, and other types of data that require different transformations.
We have **ColumnTransformer** and **make_column_transformer** and the difference is same as in pipelines i.e.
**make_column_transformer** doesn't need naming. <br> <br>
**ColumnTransformer Implementation Code :**
    ```python
    from sklearn.compose import ColumnTransformer
    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
                  "total_bedrooms", "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]
    cat_pipeline = make_pipeline(
       SimpleImputer(strategy="most_frequent"),
       OneHotEncoder(handle_unknown="ignore"))
    preprocessing = ColumnTransformer([
       ("num", num_pipeline, num_attribs),
       ("cat", cat_pipeline, cat_attribs),
    ])
    ```
  **make_column_transformer Implementation Code :**
    ```python
  from sklearn.compose import make_column_selector, make_column_transformer
  preprocessing = make_column_transformer(
     (num_pipeline, make_column_selector(dtype_include=np.number)),
     (cat_pipeline, make_column_selector(dtype_include=object)),
  )
  ```
  Here, *make_column_selector* is used to select the specific type of features in the dataset. 


- Applied almost all the transformations that we did before with the help of pipelines and column transformers with the code screenshot
attached below: <br> <br>
![pipeline_columnTransformers](Day3/pipeline_columnTransformers.png)

___

