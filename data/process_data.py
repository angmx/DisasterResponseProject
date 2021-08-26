# Author: Angelina Espinoza-Limón
# Project: DisasterResponse
# File for loading, cleaning and saving data from files to a database

import sys
import pandas as pd
import numpy as np
import seaborn as sns
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import re


# Function to extract the column names by extracting the text before the hyphen in the column categorie
def colsName(colnames):
    """
    This function obtain the column names with the text before the hyphen in each colnames entry
    
    Parameters: colname is the string from the text before the hyphen is obtained
    Return:     category_colnames is the list with columns names for the message categories
    """
    category_colnames=[] # Stores the messages categories as colunm names

    # select the first row of the categories dataframe
    row = colnames
    #print("row=",row)

    cols=re.split(';',row)
    #print("\n",cols)

    for x in cols:
        col=re.split("-",x)[0] # Get the text before the hyphen
        category_colnames.append(col)

    return category_colnames

def ConvertColumn(categories):
    """
    This function obtain the column names by taking the text before the hyphen in each colnames entry
    
    Parameters: colname is the string from the text before the hyphen is obtained
    Return:     category_colnames is the list with columns names for the message categories
    """
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x.strip()[-1] for x in categories.loc[:,column]]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    return categories


# Function that extracts the category names and create splitted columns with the categories column
def ReplaceCategories(df, categories):
    """
    This function to extract the category names and create splitted columns with the categories column
    
    Parameters: df is dataframe which will replace the category columns with columns per category
                categories is the dataframe with the categories
    Return:     df is the dataframe with categories as columns
    """
    # create a dataframe of the 36 individual category columns
    s = pd.Series(df.categories)
    #print(s.head())
    
    s_categories = s.str.split(pat=';',expand=True)   # Obtain the categories names
    s_categories.head()
    
    # Get the columns names by categorie
    category_colnames = colsName(categories.iloc[0][1]) 
    #print("\ncategory_colnames=",category_colnames) # Statement given 
 
    categories = pd.concat([categories, s_categories], axis=1)
    
    # Drop the original categories column
    categories.drop(columns=['categories','id'], inplace=True)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert the categories columns to the last integer in the category-string
    categories = ConvertColumn(categories)
    print("Categories in the dataframe=",categories.columns)
    print("Categories head=",categories.head())
    
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1, inplace=True)
    #df.head()

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    #df.head()
    return df


def load_data(messages_filepath, categories_filepath):
    """
    This function loads data from the provided files
    
    Parameters: messages_filepath is the file with the messages
                categories_filepath is the file with the categories
    Return:     df is the dataframe with messages and categories
    """
 
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    #print(messages.columns)
    #messages.head()
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #print(categories.columns)
    #categories.head()
    
    df = messages.merge(categories, how='outer', on=['id'])
    #df.head()
    
    df = ReplaceCategories(df, categories)
    #df.head()
    
    return df


def clean_data(df):
    """
    This function cleans data from the provided dataframe
    
    Parameters: df is the dataframe with the messages and categories
    Return:     df is the dataframe already clean with no duplicates
    """

    print("df shape=",df.shape)

    duplicate = df[df.duplicated()]
    print("df shape of duplicates=",duplicate.shape )
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    # check number of duplicates
    duplicate = df[df.duplicated()]
    print("df shape with no duplicates=",duplicate.shape )
    print("Dataframes columns = ", df.columns)    
    print("Dataframe head=",df.head())    

    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    return df


def save_data(df, database_filename):
    """
    This function saves the data to a database
    
    Parameters: df is the dataframe with the messages and categories
                database_filename is the name of the output database
    Return:     None
    """
    from sqlalchemy import schema
    from sqlalchemy.schema import DropTable, DropConstraint
    from sqlalchemy import inspect
    
    database_name = "sqlite:///"+ database_filename # Get the database name from the IO
    print(database_name)

    table_name = re.split('.db', database_filename) # Get the table name from the IO
    print(table_name)

    engine = create_engine(database_name)  # Create engine for database
    print(engine)
    conn = engine.connect()                # Get the conecctions to the database
    
    
    df.to_sql(table_name[0], conn, index=False, if_exists='replace') # Create the table and save the dataframe to that table, if exists then replace the table
    pass  


def main():
    """
    This function loads, cleans and saves data to a database
    
    Parameters: messages_filepath is the file with the messages
                categories_filepath is the file with the categories
                database_filepath is the name of the output database
    Return:     None
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