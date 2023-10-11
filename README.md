# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


Disaster Response Pipeline
Overview

The Disaster Response Pipeline is a Python project that provides a robust solution for processing and categorizing messages related to disaster response. It includes two main components: data processing (process_data.py) and machine learning model training (train_classifier.py). This pipeline is designed to assist emergency responders in efficiently identifying and categorizing messages during a crisis.
Key Features

    Data Loading and Merging: The process_data.py script loads and merges messages and categories data from CSV files into a single DataFrame.

    Data Cleaning and Preprocessing: The clean_data(df) function in process_data.py performs data cleaning tasks, including splitting categories, converting values to binary, dropping duplicates, and filtering out irrelevant rows.

    Data Saving: The cleaned data is saved to an SQLite database for easy access and retrieval.

    Machine Learning Model: The train_classifier.py script defines a machine learning pipeline that tokenizes text, performs TF-IDF transformation, and uses a Random Forest classifier for multi-output classification.

    Model Evaluation: Classification reports are generated to evaluate the model's performance for each category, providing precision, recall, and F1-score metrics.

    Command-Line Interface: Both scripts can be executed from the command line, making it easy to specify input and output filepaths.


Contents

    |-- README.md
    |-- app:
        |-- templates:
            |-- go.html
            |-- master.html
        |-- run.py
    |-- data:
        |-- DisasterResponse.db
        |-- disaster_categories.csv
        |-- disaster_messages.csv
        |-- process_data.py
    |-- modals:
        |-- classifier.pkl
        |-- train_classifier.py



Usage

    Clone this repository to your local machine.
    Navigate to the project directory.
    Execute python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db to process and store data in a SQLite database.
    Execute python train_classifier.py DisasterResponse.db classifier.pkl to train the machine learning model and save it as a pickle file.
    Customize and integrate the model into your disaster response system for real-world application.

Documentation

Detailed documentation and function descriptions are provided in the code with docstrings to guide users in understanding and using the pipeline.
Dependencies

    Python 3.x
    Libraries: sys, pandas, sqlalchemy, nltk, scikit-learn

Author

AlKhurayyif
License



