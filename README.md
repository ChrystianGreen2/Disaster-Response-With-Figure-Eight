# Disaster Response Pipeline Project
### Overview
In this project we used a dataset with disaster responses from Figure Eight, applying Data Engineering Techniques to Extract data from various sources, transform them in a single datasource and load in a Machine Learning Model.

In this project we also applyied NLP techniques do clean data before input on models, after that we build a model pipeline with GridSearch to hyperparameter tuning and finnaly save the model to use in a web application.


Dependencies
~~~~~~~~~~~~
- Python
- Pandas
- Sklearn
- Nltk
- sqlalchemy
~~~~~~~~~~~~

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
