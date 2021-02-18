import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """Loads CSV files into pandas DataFrame objects and merges on 'id' column
    to return single dataframe.
    
    Args:
        messages_filepath (string): file to read messages data from
        categories_filepath (string): file to read categories data from
    
    Returns:
        pandas DataFrame: merged dataframe of messages and categories data
    
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')
    return df

def clean_data(df):
    
    """Splits 'categories' column into separate columns per individual category,
    replaces category values with numeric binary, and removes duplicate rows.
    
    Args:
        df (pandas DataFrame): data to clean
    
    Returns:
        pandas DataFrame: cleaned dataframe
    
    """
    
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = [category[:category.index('-')] for category in categories.iloc[0].values.tolist()]
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    df[df.columns[4:]] = df[df.columns[4:]].replace(2, 1)
    return df

def save_data(df, database_filepath):
    
    """Loads cleaned data into SQLite database.
    
    Args:
        df (pandas DataFrame): data to loads
        database_filepath (string): filepath to save data to
    
    Returns:
        None
    
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
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