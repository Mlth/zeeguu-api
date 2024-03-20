from typing import Union
import pandas as pd

def __initial_session_setup(size) -> Union[pd.DataFrame, list]:
    '''
    Intial setup
    You can change the size of the users and articles by changing the size parameter, this will change it for all mocks
    it returns a Union of users with id and name an empty list of sessions_data 
    '''
    users = pd.DataFrame()
    users['id'] = range(0, size)
    users['name'] = [f'user_{i}' for i in range(0, size)]
    sessions_data = []
    return users, sessions_data

def setup_session_5_likes_range(num_users, num_items) -> pd.DataFrame:
    users, sessions_data = __initial_session_setup(num_users)
    for user_id in users['id']:
        # Determine the range of article IDs based on user ID
        min_article_id = user_id - 5
        if(min_article_id < 0):
            min_article_id = 0
        max_article_id = user_id + 5
        if(max_article_id > num_items-1):
            max_article_id = num_items
        liked_articles = [article_id for article_id in range(min_article_id, max_article_id)]        
        # Append user ID, article IDs within the range, and the expected read value to sessions_data
        for article_id in liked_articles:
            if(article_id != 50): #This is to show that everyone gets 50 recommended because none likes it 🤔🤔🤔
                sessions_data.append({'user_id': user_id, 'article_id': article_id, 'expected_read': 1.0})
    # Create a DataFrame from sessions_data
    sessions = pd.DataFrame(sessions_data)
    return sessions

def generate_articles_with_titles(size) -> pd.DataFrame:
    '''This mocks 100 articles in a DataFrame'''
    articles = pd.DataFrame()
    articles['id'] = range(0, size)
    articles['title'] = [f'article_{i}' for i in range(0, size)]
    return articles