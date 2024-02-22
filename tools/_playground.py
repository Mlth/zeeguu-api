import sys

from elasticsearch import Elasticsearch
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage, UserActivityData, UserArticle
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

    dot_color = np.where(df['liked'] == 1, 'blue', 'red')
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
    plt.text(0, 1.1, f"Have liked: {have_read_ratio:.2f}%", transform=plt.gca().transAxes)
    plt.text(0, 1.05, f"Have not liked: {have_not_read_ratio:.2f}%", transform=plt.gca().transAxes)

    plt.savefig(file_name + '.png')
    print("Saving file: " + file_name + ".png")
    plt.show()

def switch_difficulty(number):
    result = 0

    if 0 <= number <= 10:
        result = 1
    elif 11 <= number <= 20:
        result = 2
    elif 21 <= number <= 30:
        result = 3
    elif 31 <= number <= 40:
        result = 4
    elif 41 <= number <= 50:
        result = 5
    elif 51 <= number <= 100:
        result = 6

    return result

def get_all_user_reading_sessions():
    sessions = {}
    liked_sessions = []
    have_read_sessions = 0

    query_data = (
        UserReadingSession.query
            #.filter_by(user_id = 534)
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
        liked = UserArticle.query.filter_by(user_id=user_id, article_id=article_id).with_entities(UserArticle.liked).first()
        liked_value = 0 if liked == (False,) or liked is None else 1 # should check out
        
        if (user_id, article_id) not in sessions:
            sessions[(user_id, article_id)] = {
                'user_id': user_id,
                'article_id': article_id,
                'user_duration': session_duration,
                'language': article.language_id,
                'difficulty': article.fk_difficulty,
                'word_count': article.word_count,
                'liked': liked_value,
                'days_since': (datetime.now() - session.start_time).days,
            }
        else:
            sessions[(user_id, article_id)]['user_duration'] += session_duration
    
    for session in sessions.keys():
        user_id = session[0]
        article_id = session[1]

        should_spend_reading_lower_bound = get_expected_reading_time(sessions[session]['word_count'], 20)
        should_spend_reading_upper_bound = get_expected_reading_time(sessions[session]['word_count'], -20)

        user_level = UserLanguage.query.filter_by(user_id = user_id, language_id=sessions[(user_id, article_id)]['language']).filter(UserLanguage.cefr_level.isnot(None)).with_entities(UserLanguage.cefr_level).first()
        if user_level is None or user_level[0] == 0 or user_level[0] is None or user_level[0] == [] or user_level == []:
            usr_lvl = 1
        else:
            usr_lvl = user_level[0]
        
        #print(sessions[(user_id, article_id)]['language'], usr_lvl)
      


        diff = sessions[session]['difficulty']


        calcDiff = switch_difficulty(diff)
        
        
        if calcDiff > usr_lvl:
            diff = 1 + ((calcDiff - usr_lvl) / 10)
        elif calcDiff < usr_lvl:
            diff = 1 - ((usr_lvl - calcDiff) / 10)
        else:
            diff = 1
    
        

        timesTranslated = translated_words_per_article(user_id, article_id)
        userDurationWithTranslated = (sessions[session]['user_duration'] - (timesTranslated * 3)) # * diff
        sessions[session]['user_duration'] = userDurationWithTranslated 
        
        if userDurationWithTranslated <= should_spend_reading_upper_bound and userDurationWithTranslated >= should_spend_reading_lower_bound and sessions[session]['liked'] == 0:
            have_read_sessions += 1
            sessions[session]['liked'] = 1
        elif sessions[session]['liked'] == 1:
            have_read_sessions += 1
            sessions[session]['haveRead'] = 1
            liked_sessions.append(sessions[session])
        else:
            continue

    #df = pd.DataFrame.from_dict(sessions, orient='index')

    return sessions, liked_sessions, have_read_sessions

def session_map_to_df(sessions):
    df = pd.DataFrame.from_dict(sessions, orient='index')
    return df

def session_list_to_df(sessions):
    df = pd.DataFrame(sessions, columns=['user_id', 'article_id', 'user_duration', 'language', 'difficulty', 'word_count', 'haveRead', 'days_since'])
    return df

def get_dfs():
    sessions, liked_sessions, have_read_sessions = get_all_user_reading_sessions()

    df = session_map_to_df(sessions)
    liked_df = session_list_to_df(liked_sessions)

    return df, liked_df, have_read_sessions

def plot_df(df, have_read_sessions):
    plot_urs_with_duration_and_word_count(df, have_read_sessions, "test")

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

#Getting and plotting the data frames. We plot the df with all values (also non-observed), since we want the graph to show all user-article pairs (user_reading_sessions)
df, liked_df, have_read_sessions = get_dfs()
plot_df(df, have_read_sessions)

# We build the tensor with only the data frame containing the liked articles. This is because the tensor should be sparse, and only contain the interesting data.
tensor = build_sparse_tensor(liked_df)
print_sparse_tensor(tensor)

print("after test")