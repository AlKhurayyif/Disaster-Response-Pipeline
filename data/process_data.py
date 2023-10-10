import sys
from sqlalchemy import create_engine 
import pandas as pd


def load_data(messages_filepath, categories_filepath):
     # Read messages and categories
     messages = pd.read_csv(messages_filepath)
     categories = pd.read_csv(categories_filepath)
    
     # Merge the two dataframes
     df = messages.merge(categories, on='id')
     return df


def clean_data(df):
      # Split categories into different columns
     categories = df['categories'].str.split(';', expand=True)
    
     # Extract column names from the first row
     row = categories.iloc[0]
     category_colnames = row.apply(lambda x: x.split('-')[0])
     categories.columns = category_colnames
    
     # Convert category values to 0 or 1
     for column in categories:
         categories[column] = categories[column].str[-1].astype(int)
    
     # Drop the original categories column from df
     df = df.drop(columns=['categories'])
    
     # Concatenate the original dataframe with the new categories dataframe
     df = pd.concat([df, categories], axis=1)
    
     # Drop duplicates
     df = df.drop_duplicates()
    
     # Drop rows with 'related' column equal to 2
     df = df[df['related'] != 2]
    
     # Convert 'id' column to string type
     df['id'] = df['id'].astype(str)
    
     return df   

    
def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')

    
def main():
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