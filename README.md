# Disaster Response Pipeline
NLP classification model to categorise disaster response messages via an interactive web app.

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Data Preprocessing and Modeling](#model)
4. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python version 3.

## Project Motivation<a name="motivation"></a>

The motivation behind undertaking this project was to gain familiarity with the process of developing an ETL pipeline to train a machine learning model. The steps of the process were as follows:

1. Creating a file to extract and transform labaled disaster response messages from CSV files and load the data into a SQLite database
2. Developing and tuning a classification model and saving the model into a [pickle](https://docs.python.org/3/library/pickle.html) file
3. Creating a [Flask](https://flask.palletsprojects.com/en/1.1.x/) app to display [Plotly](https://plotly.com/) visualisations of the training data and provide an interface for predicting the labels of new messages
4. Writing the HTML files to display the app as a web page using [Bootstrap](https://getbootstrap.com/) templates

The version of the files contained in this repository can be used to render the app on a local machine.

## Data Preprocessing and Modeling<a name="model"></a>

The data for this project was provided in two separate CSV files: one containing the messages and the other containing their binary classifications of 36 category labels.

After merging the data sets, the steps for processing the messages into a model-friendly format were as follows:

1. Remove all punctuation and special characters from the text using a regular expression
2. Tokenise each document in the text
3. Lemmatise tokens, set to lower case, strip whitespace and filter out stop words
4. Convert documents into a matrix of vectorised token counts
5. Transform matrix into tf-idf representation

This process rendered the messages into a matrix of fearures suitable to train a Random Forest Classification model, with the 36 categories acting as a multilabel target variable.

The NLP preprocessing and model fitting were bundled into a single pipeline to better facilitate the saving of the model to then be loaded in the web app.

## File Descriptions <a name="files"></a>

- app: folder containing HTML files for web page and run.py file to load the data, create the visualisations and deploy the model
- data: folder containing CSV files of the original data, data_processing.py file containing ETL pipeline, and final data set loaded into SQLite database
- models: folder containing file to train and save model and final model in classifier.pkl

To render the dashboard locally, download and navigate to the app folder of the repository and enter the command

`
python run.py
`

## Licensing, Authors, Acknowledgements <a name="licensing"></a>

The data for this project was provided by Figure Eight (now [Appen](https://appen.com/))
