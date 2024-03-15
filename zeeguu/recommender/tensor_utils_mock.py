import tensorflow as tf
import pandas as pd
import random
import numpy as np


def build_mock_sparse_tensor():
    users = setup_users()
    articles = genereate_100_articles_with_titles()
    sessions = setup_sessions(users, articles)

    print("Building mock tensor")

    # Sort the indices
    sessions = sessions.sort_values(['user_id', 'article_id'])
    indices = sessions[['user_id', 'article_id']].values
    values = sessions['expected_read'].values

    tensor = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[len(users), len(articles)]
    )

    # Convert sparse tensor to dense tensor
    dense_tensor = tf.sparse.to_dense(tensor)

    # Print the tensor within TensorFlow session
    with tf.compat.v1.Session() as sess:
        tensor_value = sess.run(dense_tensor)
        print(tensor_value)

    return tensor

def setup_users() -> pd.DataFrame:
    '''This mocks 100 users in a DataFrame'''
    users = pd.DataFrame()
    users['id'] = range(0, 100)
    users['name'] = [f'user_{i}' for i in range(0, 100)]
    return users

def setup_articles() -> pd.DataFrame:
    '''This mocks 100 articles in a DataFrame'''
    articles = pd.DataFrame()
    articles['id'] = range(0, 100)
    articles['title'] = [f'article_{i}' for i in range(0, 100)]
    return articles

def setup_sessions(users, articles) -> pd.DataFrame:
    '''This mocks sessions in a DataFrame where each user likes 5 articles'''
    sessions_data = []
    for user_id in users['id']:
        liked_articles = random.sample(articles['id'].tolist(), 15)
        for article_id in liked_articles:
            sessions_data.append({'user_id': user_id, 'article_id': article_id, 'expected_read': 1.0})
    sessions = pd.DataFrame(sessions_data)
    return sessions

def genereate_100_articles_with_titles() -> pd.DataFrame:
    '''This mocks 100 articles in a DataFrame'''
    articles = pd.DataFrame()
    articles['id'] = range(0, 100)
    articles['title'] = [f'article_{i}' for i in range(0, 100)]
    return articles