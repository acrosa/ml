
# coding: utf-8

# In[62]:


import tensorflow as tf
import numpy as np
import pandas as pd


# In[63]:


import_data = pd.read_csv("datasets/imports/imports.csv", skipinitialspace=True, skiprows=1, na_values="?")

# In[64]:


# set missing column names
import_data.columns = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length", 
    "width", 
    "height", 
    "curb-weight", 
    "engine-type",
    "num-of-cylinders",
    "engine-size", 
    "fuel-system",
    "bore", 
    "stroke", 
    "compression-ratio", 
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price"
]


# In[65]:


# split train data into validation data
train_data = import_data.sample(frac=0.7, replace=False)
validation_data = import_data[len(train_data):]

# cleanup data
train_data = train_data[np.isfinite(train_data['price'])]


# In[66]:


train_data.head()


# In[68]:


# build train input function
input_fn_train = tf.estimator.inputs.pandas_input_fn(
    x = pd.DataFrame({
        "make" : train_data["make"].values,
        "highway-mpg" : train_data["highway-mpg"].values,
        "curb-weight" : train_data["curb-weight"].values,
        "body-style" : train_data["body-style"].values
    }),
    y = pd.Series(train_data["price"].values),
    shuffle=True,
    batch_size=128
)

feature_columns = [
    tf.feature_column.numeric_column("highway-mpg"),
    tf.feature_column.numeric_column("curb-weight"),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(key="body-style", vocabulary_list=[
            "hardtop", "wagon", "sedan", "hatchback", "convertible"
        ])
    ),
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(key="make", hash_bucket_size=50),
        dimension=3
    )
]
estimator = tf.estimator.DNNRegressor(feature_columns=feature_columns, hidden_units=[20, 20])
estimator.train(input_fn=input_fn_train, steps=5000)


# In[71]:


input_fn_validation = tf.estimator.inputs.pandas_input_fn(
    x = pd.DataFrame({
        "make" : validation_data["make"].values,
        "highway-mpg" : validation_data["highway-mpg"].values,
        "curb-weight" : validation_data["curb-weight"].values,
        "body-style" : validation_data["body-style"].values
    }),
    y = pd.Series(validation_data["price"].values),
    shuffle=True,
    batch_size=128
)

eval_result = estimator.evaluate(input_fn=input_fn_validation, steps=100)

average_loss = eval_result["average_loss"]

# Convert MSE to Root Mean Square Error (RMSE).
print("\n" + 80 * "*")
print("\nRMS error for the test set: ${:.0f}"
    .format(average_loss**0.5))

