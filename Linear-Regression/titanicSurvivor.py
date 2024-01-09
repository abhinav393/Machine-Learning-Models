import pandas as pd
import tensorflow as tf

'''
    The dataset I will be focusing on here is the titanic dataset. 
    It has tons of information about each passenger on the ship. 
    Our first step is always to understand the data and explore it. So, let's do that!
    Below we will load a dataset and explore it using some built-in tools.
'''

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data

'''
    The pd.read_csv() method will return a new pandas dataframe. You can think of a dataframe like a table.
    
    I loaded two different datasets above. 
    This is because when we train models, we need two sets of data: training and testing.

    The training data is what I feed to the model so that it can develop and learn. 
    It is usually a much larger size than the testing data.

    The testing data is what I use to evaluate the model and see how well it is performing. 
    I must use a seperate set of data that the model has not been trained on to evaluate it. Can you think of why this is?

    Well, the point of our model is to be able to make predictions on NEW data, data that we have never seen before. 
    If we simply test the model on the data that it has already seen we cannot measure its accuracy accurately. 
    We can't be sure that the model hasn't simply memorized our training data. This is why we need our testing and training data to be separate.
'''

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

'''
    I've decided to pop the "survived" column from our dataset and store it in a new variable. 
    This column simply tells us if the person survived our not.

    I can actually have a look at the table representation.
    To look at the data I'll use the .head() method from pandas. This will show me the first 5 items in the dataframe.
'''

print(dftrain.head())

# And if we want a more statistical analysis of our data we can use the .describe() method.

print(dftrain.describe())

print(dftrain.shape)  # let's have a look at that too!

'''
    So I have 627 entries and 9 features, nice!
    Now let's have a look at the survival information.
'''

print(y_train.head())

'''
    Notice that each entry is either a 0 or 1. Can you guess which stands for survival?
    And now because visuals are always valuable let's generate a few graphs of the data.
'''

print(y_eval.head())

'''
    In this dataset, I have two different kinds of information: Categorical and Numeric
    The categorical data is anything that is not numeric! For example, the sex column does not use numbers, 
    it uses the words "male" and "female".
    Before I continue and create/train a model we must convert our categorical data into numeric data. 
    We can do this by encoding each category with an integer (ex. male = 1, female = 2).

    Fortunately for us TensorFlow has some tools to help!
'''

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

'''
    Let's break this code down a little bit...

    Essentially what we are doing here is creating a list of features that are used in our dataset.
    The cryptic lines of code inside the append() create an object that our model can use to map string values like "male" and "female" to integers. 
    This allows us to avoid manually having to encode our dataframes.

    And here is some relevant documentation

    https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list?version=stable
'''

print(feature_columns)

'''
    So, we are almost done preparing our dataset and I feel as though it's a good time to explain how our model is trained. 
    Specifically, how input data is fed to our model.
    For this specific model data is going to be streamed into it in small batches of 32. 
    This means we will not feed the entire dataset to our model at once, but simply small batches of entries. 
    We will feed these batches to our model multiple times according to the number of epochs.
    An epoch is simply one stream of our entire dataset. 
    The number of epochs we define is the amount of times our model will see the entire dataset. 
    We use multiple epochs in hope that after seeing the same data multiple times the model will better determine how to estimate it.

    Ex. if we have 10 ephocs, our model will see the same dataset 10 times.

    Since we need to feed our data in batches and multiple times, we need to create something called an input function. 
    The input function simply defines how our dataset will be converted into batches at each epoch.
    
    The TensorFlow model we are going to use requires that the data we pass it comes in as a tf.data.Dataset object. 
    This means we must create a input function that can convert our current pandas dataframe into that object.
    
    Below you'll see a seemingly complicated input function, 
    this is straight from the TensorFlow documentation (https://www.tensorflow.org/tutorials/estimator/linear). 
    I've commented as much as I can to make it understandble, 
    but you may want to refer to the documentation for a detailed explaination of each method.
'''


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices(
            (dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(
            num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use


train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
'''
    we are going to use a linear estimator to utilize the linear regression algorithm.
    we create the linear estimator by passing the feature columns we created earlier.
'''

linear_est.train(train_input_fn)  # train

result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

print(result['accuracy'])  # the result variable is simply a dict of stats about our model

result = list(linear_est.predict(eval_input_fn))  # predicts for us
print(result)
print(result[0])
print(result[0]['probabilities'])  # for both surviving and not surviving of person at 0
print(result[0]['probabilities'][0])  # for not surviving of person at 0
print(result[0]['probabilities'][1])  # for surviving of person at 0

print(dfeval.loc[0])  # for getting stats of person at 0
print(y_eval.loc[0])  # for confirming the probability is right or not
