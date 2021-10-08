# Turning Machine Learning Models into products with Flask

Build an application for your model using REST APIs with Flask


*Read the original post [here]*



# Introduction


In 1971, with just 21 years old, Steve Wozniak was already an incredible computer programmer and elecronics engineer. He really enjoyed to spend his time creating weird devices and searching for improvements in his area. It was in 1971 when he met a 15 year old guy who would soon realize the enormous potential of what Wozniak was doing. This guy thought that the weird devices Wozniak created could become useful for many people. They worked together the following years, and 6 years later, they started to commercialize one of the first and most successful personal computers of the era: the Apple II.

The guy that convinced Wozniak to convert his devices into products was Steve Jobs. It is curious, because everybody knows who Steve Jobs is, but not many people have heard about Wozniak.


This post is not about the story behind Apple, but it is about turning technology into products.




When you spend a lot of time with notebooks and competitions, it seems that the Machine Learning picture is pretty simple: you create models that throw a metric. This metric explains how good your model is, so you have to improve the metric. Sometimes you have a threshold. Eventually you achieve this threshold. And once you achieve it, it means that the model is good enough so, job is done!

It turns that in a real environment this process is just the tip of the iceberg. We have to remember that at the end of the day, what we want to do is to create products that are useful for the people.

In this post I will develop a web application for a Machine Learning model. I will use Flask, a microframework where you can build a REST API in a simple way with Python code. 


We will work with [Adult Census Income](https://www.kaggle.com/uciml/adult-census-income) Dataset. We will try to predict if a person earns more than 50k or not, thus we have a binary classification problem. As explanatory variables, we will use personal characteristics as age, sex, marital status and working hours per week. 

We will use HTML for building the web application. I know many people reading this are not familiar with it, so if you are one of them, don't worry! I am not an expert either so the code here is not a big deal. It is also worth mentioning that in this post I will focus on developing a REST API, so the web development will take a secondary place.


this post is structured as follows: First we will define in an easy way the concepts we have to know before building a REST API. Then, I will explain the code step by step. You can find all the code used for this proyect in [this repository](https://github.com/Adricarpin/Flask-REST-API.git).


That being said, let's get to the point!


# Key concepts behind REST API


To make things clear, I think first we have to answer some typical questions about REST APIs in order to understand what a REST API really is. These are short questions and anwers that shouldn't be a big deal to understand even if you are not familiar with the concept of REST API.

- **What is an API?**

API is the acronym for Application Programming Interface, which is basically software intermediary that allows two applications to talk to each other.

- **What is REST?**

REST can be defined as set of rules that developers follow when they create their API. It  stands for REpresentational State Transfer, and is a stateless architecture that generally runs over HTTP.


- **What is the difference between REST and RESTful?**

While REST refers to the arquitecture, RESTful refers to a web service that implements the REST architecture.

- **How a RESTful API works?**

What a RESTful API essentially does is send requests to obtain resources.

- **What is exactly a request?**

When you make a request in HTTP you are asking the server for something. 
For doing a request you use HTTP methods. The most important methods are the following:

```GET``` to fetch data.

```PUT``` to alter the state of data.

```POST``` to create data.

```DELETE``` to remove data.

As an example, when you go to http://www.google.com, you send the following:

```
GET / HTTP/1.1
Host: www.google.com
```

This is a GET request where:

1. ```GET``` is the verb
2. ```/``` is the path
3. ```HTTP/1.1``` is the protocol


- **What is exactly a resource?**

A resource is essentially data. For example, if you ask for your model predictions, the predictions, than can be some sort of data like a JSON file, will be the resource.


I think that with this key concepts in mind we are now ready to build a REST API with Flask, so let's get down to business!


# REST API implementation with Flask


Before going with the code, I think is essential to first set the plan:

Essentially, we want to create an application that takes as **inputs** the different characteristics of and individual, and **outputs** the probability that the individual has to earn more than 50k given those characteristics.

Therefore, in the main page we will need to specify the inputs:

![index_full](https://user-images.githubusercontent.com/86348959/136525948-db643913-a4f4-4d32-97ae-da989aa03b88.png)


and once we submit them, the application will send us to a page where we can see the outputs:


![output_full](https://user-images.githubusercontent.com/86348959/136525967-928012a3-6e96-4ab7-8f05-03b4e1b58610.png)


The main files in the [repository](https://github.com/Adricarpin/Flask-REST-API.git) are the Python scripts. In ```model.py``` we will build our model while in ```server.py``` we will build the REST API.

I  explain both scripts below.



## Building the model


The first step is to create a model that makes predictions. For doing that, we first process the data:


```python
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


# Read data
df = pd.read_csv('adult.csv', sep=",")


# Drop useless variables
df = df.drop(['fnlwgt','education.num', 'occupation', 'relationship', 
        'capital.gain', 'capital.loss', 'native.country'], axis=1)


# Set income as a binary variable
df['income'].replace(['<=50K','>50K'],[0,1], inplace=True) 


# delete rows with missing data
df = df.loc[df['workclass'] != '?']


# Split into dependend and independent variables
X = df.drop('income', axis=1)
y = df['income']


# Split X into continous variables and categorical variables
X_continous  = X[['age', 'hours.per.week']].reset_index(drop=True)

X_categorical = X[['workclass', 'education', 'marital.status',  'race',
                   'sex']].reset_index(drop=True)


# Fit One hot encoder
enc = OneHotEncoder()
enc.fit(X_categorical)


# categorical data to One hot encoding
X_encoded = enc.transform(X_categorical).toarray()
X_encoded = pd.DataFrame(X_encoded)


# Concatenate both continous and encoded sets:
X = pd.concat([X_continous, X_encoded], axis=1)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,
                                                    stratify=y,random_state=10 )

```

Once we have the processed data, we can fit a model. I have chosen a logistic regression (it works pretty well). 


```python
# Model
logit = LogisticRegression(max_iter=10000)
logit = logit.fit(X_train, y_train)
```

:ledger:NOTE: if you want to develop more complex models for this problem, you can take a look at this [Kaggle notebook](https://www.kaggle.com/adro99/from-na-ve-to-xgboost-and-ann-adult-census-income).

 
Once we have our model fitted, we have to create a function that does the following:

It takes as inputs single values for each of the explanatory variables, i.e., characteristics of an individual.

It **outputs** the probability the individual has to earn more than 50k given their characteristics (inputs).


```python
def predict_probability(age, workclass, education, marital_status, race, sex, hours):
    """This function predicts the probability of earning more than 50K a year, given 
    some input variables"""

    encoded = enc.transform([[workclass, education, marital_status, race, sex]]).toarray()
    encoded = encoded.reshape(encoded.shape[1],)
    continous = np.array([age, hours])

    processed_data = np.concatenate((continous, encoded))
    processed_data = processed_data.reshape(1, processed_data.shape[0])
    prediction = logit.predict_proba(processed_data)[0][1]
    prediction = "{}%".format(round(prediction*100, 2))

    return prediction 
```

As you may have noticed, our web application has the same inputs and the same output, thus this function is key for this project.


## Building the REST API


In the script ```server.py``` we will develop the REST API.

We first import some libraries and create intances of the ```Flask``` class and the ```Api``` class.

```python
import model 
from flask import Flask, request, render_template, make_response
from flask_restful import Resource, Api


app = Flask(__name__)
api = Api(app)
```


Then we will create 2 classes. Both of them inherit the ```Resource``` class and both of them are composed of a function called ```get```. If you remember, at the beginning we said that what a RESTful API basically does is use requests to obtain resources. As you might guess, the functions called ```get``` are requests that uses the GET method, and both of them will return resources.

The first one is pretty easy: it basically returns the HMTL code used to make the application page where we set the inputs.

```python
class index(Resource):
    def get(self):
        return make_response(render_template('index.html'))
```


The second one gets the inputs defined by us in the web application and passes them to the ```predict_probability``` function that returns a probability. Finally, the ```get``` function returns the HTML code used to make the application page where the probability is showed.

```python
class probability(Resource):
    def get(self):
        try:
            age = int(request.args.get('age'))
            workclass = str(request.args.get('work'))
            education = str(request.args.get('education'))
            marital_status= str(request.args.get('status'))
            race = str(request.args.get('race'))
            sex = str(request.args.get('sex'))
            hours = int(request.args.get('hours'))

            prediction = model.predict_probability(age, workclass, education, marital_status, race, sex, hours)

            return make_response(render_template('output.html', prediction=prediction))

        except:
            return 'An Error occurred'
```


Then we have to add the classes as resources. For doing that, we use the ```add_resource``` method, specifying the name of the class and the URL where they will go.

```python
api.add_resource(index, '/')
api.add_resource(probability, '/probability')
```


At the end of the script, we run the app.

```python
if(__name__=='__main__'):
    app.run(debug=True)  
```


:ledger:NOTE: If you don't know what the ```if(__name__=='__main__')``` does, here is an explanation: Python assigns the name ```__main__``` to a script when it is executed. If the script is imported from another script, the script keeps its given name (not ```__main__```). Therefore, using the ```if(__name__=='__main__')``` statement we make sure that the app is only run when we are executing the ```server.py``` script. Thus,  if we are on an other script and we import something from the server.py script, the app doesn't run. This is a common practice for avoiding undesired executions of code.




Besides the Python scripts, the repository also has other files that are worth mentioning:

- ```adult.csv```

This is the raw data. You can also access it in [Kaggle](https://www.kaggle.com/uciml/adult-census-income).

- ```requirements.txt```

In is file you can see the requirements for running the code. It is not mandatory to have the exactly same versions.

- ```templates``` folder

Here I store the HTML code for building the web application. ```index.html``` has the code for building the main page (where we set the inputs) and ```output.html``` has the code for building the output page (where we see the probability). 

:ledger:NOTE: I have a limited knowledge of HTML, so I am sure that the code can be improved in many ways. 

- ```static``` folder

Here I store CSS code. 

:ledger:NOTE: If you haven't heard of CSS, it is used to define styles for your web application, so it helps the HTML code to make the web application look nicer.


# Test the application

Once we know all the above, we can run our application. For doing that, open a terminal in the path of the root folder where you have the ```server.py``` script. Write ```python service.py``` to run ```service.py```. 

Once you run it you should see a line like this in the output:

```sh
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

```http://127.0.0.1:5000/``` is the direction of your application. 

127.0.0.1 is the IP of the localhost and 5000 is the port. Port 5000 is the default but you can specify any other using the ```app.run()``` method in ```server.py```.

If you click on the URL, you should see the web application in your browser.
Now you can play with it and check that everything works! 


Finally, we have converted our Machine Learning model into a product that can be developed!


I hope you have learned a lot! Thanks for reading!






