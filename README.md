# Disaster Response Pipeline Project
### Overview
In this project we used a dataset with disaster responses from Figure Eight, applying Data Engineering techniques to extract data from various sources, transform them in a single datasource and load in a machine learning model.

In this project we also applied NLP techniques do clean data before input on models, after that we build a model pipeline with GridSearch to tuning the hyperparameters and finaly save the model to use in a web application.


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

### Acknowledgment
Thanks to Udacity for the incredible project that allowed me to improve my data engineering techniques.
Thanks to Figure Eight for providing the database that was very helpful.
