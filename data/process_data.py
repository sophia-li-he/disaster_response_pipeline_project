import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    the function loads the messages file and the categories file and combines these two files together
    input: file path for the messages, file path for the categories
    output: a dataframe that combines the messages and the categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how = 'left', on = 'id')
    return df



def clean_data(df):
    """
    the function converts the category values to numbers 0 or 1 and removes duplicated data
    input: dataframe (the output of the load_data function)
    output: a cleaned dataframe with one column for each category and 0/1 indicates whehter the message belongs to the category
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # extract a list of new column names for categories
    row = categories.loc[0,]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of categories
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the column child_alone since there is one value for all the messages. 
    categories = categories.drop('child_alone', axis = 1)
    # Replace 2 as 1 in the related column
    map_related = {0:0, 1:1, 2:1}
    categories['related'] = categories['related'].map(map_related)
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    return df



def save_data(df, database_filename):
    """
    the function saves dataframe to database
    input: dataframe, string of the database filename
    output: a database file with the input filename
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)


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