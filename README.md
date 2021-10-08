# Deploy a Machine Learning Model with Flask using a REST API

Turn your code into a service that everyone understands and can interact with.

[foto captura de la app]



# Introduction


In 1971, with just 21 years old, Steve Wozniak was already an incredible computer programmer and elecronics engineer. He really enjoyed to spend the time creating weird devices and searching for improvements in his area. it was in this year when he met a 15 year old guy who would soon realize the enormous potential of what Wozniak was doing. This guy thought that the weird devices Wozniak created could become useful for many people. The following years, they worked together with this idea, and 5 years later, they created a company.

The guy that convinced Wozniak to convert his devices into products was Steve Jobs. It is curious, because everybody knows who Steve Jobs is, but not many people have heard about Wozniak.


This post is not about the story behind Apple, but it is about turning technology into products. 




When you spend a lot of time with notebooks and competitions, it seems that the Machine Learning picture is pretty simple: you create models that throw a metric. This metric explains how good your model is, so you have to improve the metric. Sometimes you have a threshold, so once you achieve it, you made it! job is done!

It turns that in a real environment this is just the tip of the iceberg.
We have to remember that at the end of the day, what we want to do is to create products that people can use. 

In this post I will develop a web application for a Machine Learning model. I will use Flask, a microframework where you can build a REST API in a simple way with Python code. 


We will work with [Adult Census Income](https://www.kaggle.com/uciml/adult-census-income) Dataset. We will try to predict if a person earns more than 50k or not, so we have a binary classification problem. As explanatory variables, we will use personal characteristics as age, sex, marital status and working hours per week. 

We will use HTML for building the web application. I know many people reading this aren't familiar with it, so if you are one of them, don't worry! I wasn't familiar with it either before I started this project! So the code here is not a big deal. It is also worth noting that in this post I will focus more on developing a REST API, as I think that's what most of you want to learn.


That being said, let's get to the point!


# teoria


To make things clear, I think first we have to answer some typical questions about REST APIs. Once we have the picture of what a REST API does, we can go with the code. 


- What is an API?

API is the acronym for Application Programming Interface, which is basically software intermediary that allows two applications to talk to each other.

- What is REST?

REST can be defined a set of rules that developers follow when they create their API. REST (REpresentational State Transfer) is basically a stateless architecture that generally runs over HTTP.


- What is the difference between REST and RESTful?
As we said REST is the arquitecture, while RESTful refers to a web service that implements the REST architecture.

- How a RESTful API works?

What RESTful API basically does is use requests to obtain resources.


- What is exactly a request?
When you make a request in HTTP you are asking the server for something. 
For doing a request you use HTTP methods. The most important methods are the following:

GET to fetch data
PUT to alter the state of data (such as an object, file, or block)
POST to create data
DELETE Remove data

As an example, when you go to http://www.google.com, you send the following:

GET / HTTP/1.1
Host: www.google.com

This is a get request

GET is the verb
/ is the path
HTTP/1.1 is the protocol


- What is exactly a resource?

A resource is essentially data. For example, if you ask for your model predictions, the predictions, than can be some sort of data like a JSON file, will be the resource.


That being said


# implementation


before explaining the code, I think is essential to first understand what is the plan. 



Essentially, we want to create an application that takes as inputs the different characteristics of and individual, and outputs the probability that the individual has to earn more than 50k given those characteristics.


[foto capturas app]

All the code for this purpose can be found in [this repository]. 


The main files in the repository are the python scripts I will explain both of them below:



## model.py


The first step is to create a model that makes predictions. For doing that, we first read and process the data.

[codigo procesamientode los datos]



the processed dataset looks like this

[foto dataset procesado]


Once we have the processed data, we can fit a model. I have chosen a logistic regression (that works pretty well). 

NOTE: if you want to develop more complex models for this problem, you can take a look at this [Kaggle notebook]


[code modelo]
 
Once we have our model fitted, we have to create a function that does the following:

It takes as inputs single values for each of the explanatory variables, i.e., characteristics of an individual (you tell it, for example, that the individual is 32 years old, is a female, works 40 hours a week...)

It outputs the probability the individual has to earn more than 50k given their characteristics (inputs).


[codigo con la funcion]


As you might guess, this function is the key of this project, as our web application has the same inputs and the same output. 



## server.py


In this script we will develop the REST API.

We first import some libraries and we create intances of the Flask class and the Api class.

[code imports / flask]


Then we create 2 classes. Both of them inherit the Resource class and both of them are composed of a function called get. If you remember, at the beginning
we said that what a RESTful API basically does is use requests to obtain resources. As you might guess, the functions called get are requests that uses the GET method, and both of them will return resources.

The first one is pretty easy: it basically returns the HMTL code used to make the application page where we set the inputs.

[code]

The second one gets the inputs defined by us in the web application and passes them to the predict_probability function thats return a probability. Finally, the get function returns the html code used to make the application page where the probability is showed.


[code]

Then we have to add the classes as resources. For doing that, we use the add_resource method, specifying the name of the class and the url where they will go.

[code]


At the end of the script, we run the app.

[code]


NOTE: If you don't know what the if(__name__=='__main__') does, here is an explanation: Python assigns the name "__main__" to a script when it is executed. If the script is imported from another script, the script keeps it given name (not __main__). Therefore, using the if(__name__=='__main__') statement we make sure that the app is only run when we are executing the server.py script (so if we are on an other script and we import something from the server.py script, the app doesn't run). This is a common practice for avoiding undesired executions.




Besides the Python scripts the repository also has other files that are worth mentioning:

- adult.csv

This is the raw data. You also access it in [Kaggle]

- requirements.txt

In is file you can see the requirements for running the code. It is not mandatory to have the exactly same versions.

- templates folder

Here I have the HTML code for building the web application. index.html has the code for building the main page (where we set the inputs) and output.html has the code for building the output page (where we see the probability). 

NOTE: I have a limited knowledge of HTML, so I am sure that the code can be improved in many ways. 

- static folder

Here I store CSS code. If you haven't heard of CSS, it is used to define styles for your web pages, so it helps the HTML code to make the web application look nicer. 



# Test the application

Once we know all the above, we can run our application. For doing that, open a terminal in the path of the root folder where you have the server.py script. Write ```python service.py``` to run service.py. 

Once you run it you shuold see a line like this in the output

 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

http://127.0.0.1:5000/ is the direction of your application. 

127.0.0.1 is the IP of the localhost and 5000 is the port. Port 5000 is the default but you can specify any other using the app.run() function from server.py.

If you click on the URL, you should see the web application in your browser.
Now you can play with it and check if everything works! 


Finally, we have converted our Machine Learning model into a product that can be developed!


I hope you have learned a lot! Thanks for reading!






