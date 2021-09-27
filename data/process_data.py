import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """load csv files and return the respectives dataframe
    Args:
        messages_filepath: the filepath messages file
        categories_filepath: the filepath of categories file
    Returns:
        df:the dataframe of messages and categories joined"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    categories = df.categories.str.split(';',expand=True)
    
    row = df.categories.values[0]
    category_colnames = list(map(lambda x: x[:-2], row.split(';')))
    categories.columns = category_colnames
    
    for i, column in enumerate(categories):
        categories[column] = categories[column].str.replace(r'[a-z_-]+','')
        categories[column] = categories[column].astype(int)
    df = df.drop(columns='categories')
    df = pd.concat([df, categories],axis=1)
    return df

def clean_data(df):
    """clean data of dataframe
    Args:
        df: dataframe
    Returns:
        df: dataframe cleaned"""
    return df.drop_duplicates()


def save_data(df, database_filename):
    """save data in database
    Args:
        df: dataframe
        database_filename: the filename of database"""
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('disaster_dataset', engine, index=False)

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