import sys
from sqlalchemy import create_engine 
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """
    Load data from message and category CSV files and merge them.

    Args:
        messages_filepath (str): Filepath to the messages CSV file.
        categories_filepath (str): Filepath to the categories CSV file.

    Returns:
        df (DataFrame): Merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Clean and preprocess the DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing messages and categories.

    Returns:
        df (DataFrame): Cleaned and preprocessed DataFrame.
    """
    categories = df['categories'].str.split(';', expand=True)
    
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    
    df = df.drop(columns=['categories'])
    
    df = pd.concat([df, categories], axis=1)
    
    df = df.drop_duplicates()
    
    df = df[df['related'] != 2]
    
    df['id'] = df['id'].astype(str)
    
    return df   

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to a SQLite database.

    Args:
        df (DataFrame): Cleaned and preprocessed DataFrame.
        database_filename (str): Filepath to the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')

def main():
    """
    Main function to load, clean, and save data to a SQLite database.

    Expects command line arguments for the messages file, categories file, and database file.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
