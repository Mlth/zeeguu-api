import sys

from elasticsearch import Elasticsearch
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage, UserActivityData
import pandas as pd
from zeeguu.core.model import db
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa # needed for pandas
from datetime import datetime, timedelta
import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

from zeeguu.api.app import create_app
from zeeguu.core.model.user_reading_session import UserReadingSession

app = create_app()
app.app_context().push()

print("Running playground")

def days_since_to_opacity(days_since):
    if days_since < 365 * 1/4:
        return 1
    elif days_since < 365 * 2/4:
        return 0.75
    elif days_since < 365 * 3/4:
        return 0.5
    elif days_since < 365:
        return 0.25
    return 0

def translated_words_per_article(user_id, article_id):
    activitySession = UserActivityData.query.filter_by(user_id=user_id, article_id=article_id).all()
    count = 0
    for activity in activitySession:
        if activity.event == "UMR - TRANSLATE TEXT":
            count += 1
    return count

def get_expected_reading_time(word_count, offset):
    # The higher the offset is, the higher we want the WPM to be. When WPM is larger, the user is expected to be able to read faster.
    # Thus, high offset/WPM = low expected reading time.
    return (word_count / (70 + offset)) * 60

def plot_urs_with_duration_and_word_count(df, have_read_sessions, file_name):
    x_min, x_max = 0, 2000
    y_min, y_max = 0, 2000

    plt.xlabel('Word count')
    plt.ylabel('Duration')

    dot_color = np.where(df['haveRead'] == 1, 'blue', 'red')
    plt.scatter(df['word_count'], df['user_duration'], alpha=[days_since_to_opacity(d) for d in df['days_since']], color=dot_color)

    x_values = df['word_count']
    y_values_line = [get_expected_reading_time(x, -20) for x in x_values]
    plt.plot(x_values, y_values_line, color='red', label='y = ')

    x_values = df['word_count']
    y_values_line = [get_expected_reading_time(x, 20) for x in x_values]
    plt.plot(x_values, y_values_line, color='red', label='y = ')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.rc('axes', axisbelow=True)

    have_read_ratio = have_read_sessions / len(df) * 100
    have_not_read_ratio = 100 - have_read_ratio
    plt.text(0, 1.1, f"Have read: {have_read_ratio:.2f}%", transform=plt.gca().transAxes)
    plt.text(0, 1.05, f"Have not read: {have_not_read_ratio:.2f}%", transform=plt.gca().transAxes)

    plt.savefig(file_name + '.png')
    print("Saving file: " + file_name + ".png")
    plt.show()

def get_all_user_reading_sessions():
    sessions = {}
    liked_sessions = []
    have_read_sessions = 0

    query_data = (
        UserReadingSession.query
            .filter_by(user_id='534')
            .filter(UserReadingSession.article_id.isnot(None))
            .filter(UserReadingSession.duration >= 30000) # 30 seconds
            .filter(UserReadingSession.duration <= 3600000) # 1 hour
            .filter(UserReadingSession.start_time >= datetime.now() - timedelta(days=365)) # 1 year
            .order_by(UserReadingSession.user_id.asc())
            .all()
    )

    for session in query_data:
        article_id = session.article_id
        user_id = session.user_id
        article = Article.find_by_id(article_id)
        session_duration = int(session.duration) / 1000 # in seconds

        if (user_id, article_id) not in sessions:
            sessions[(user_id, article_id)] = {
                'user_id': user_id,
                'article_id': article_id,
                'user_duration': session_duration,
                'language': article.language_id,
                'difficulty': article.fk_difficulty,
                'word_count': article.word_count,
                'haveRead': 0,
                'days_since': (datetime.now() - session.start_time).days,
            }
        else:
            sessions[(user_id, article_id)]['user_duration'] += session_duration
    
    for session in sessions.keys():
        user_id = session[0]
        article_id = session[1]

        should_spend_reading_lower_bound = get_expected_reading_time(sessions[session]['word_count'], 20)
        should_spend_reading_upper_bound = get_expected_reading_time(sessions[session]['word_count'], -20)

        timesTranslated = translated_words_per_article(user_id, article_id)
        userDurationWithTranslated = sessions[session]['user_duration'] - (timesTranslated * 3)
        
        if userDurationWithTranslated <= should_spend_reading_upper_bound and userDurationWithTranslated >= should_spend_reading_lower_bound:
            have_read_sessions += 1
            sessions[session]['haveRead'] = 1
            liked_sessions.append(sessions[session])
        else:
            sessions[session]['haveRead'] = 0

    return sessions, liked_sessions, have_read_sessions

def session_map_to_df(sessions):
    df = pd.DataFrame.from_dict(sessions, orient='index')
    return df

def get_df_and_plot_user_reading_sessions():
    sessions, liked_sessions, have_read_sessions = get_all_user_reading_sessions()
    df = session_map_to_df(sessions)
    liked_df = pd.DataFrame(liked_sessions, columns=['user_id', 'article_id', 'user_duration', 'language', 'difficulty', 'word_count', 'haveRead', 'days_since'])
    plot_urs_with_duration_and_word_count(df, have_read_sessions, "test")
    return df, liked_df

def build_sparse_tensor(df):
    indices = df[['user_id', 'article_id']].values
    values = df['haveRead'].values
    num_of_users = User.num_of_users()
    num_of_articles = Article.num_of_articles()
    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[num_of_users, num_of_articles]
    )

def print_sparse_tensor(tensor):
    print(tf.sparse.to_dense(tensor))

print("before the function")

df, liked_df = get_df_and_plot_user_reading_sessions()
tensor = build_sparse_tensor(liked_df)
print_sparse_tensor(tensor)

print("after test")