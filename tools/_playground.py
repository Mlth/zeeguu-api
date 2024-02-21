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

from zeeguu.api.app import create_app
from zeeguu.core.model.user_reading_session import UserReadingSession

app = create_app()
app.app_context().push()

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

def get_all_user_reading_sessions():
    sessions = {}
    have_read_sessions = 0

    query_data = (
        UserReadingSession.query
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

        #user_level = UserLanguage.query.filter_by(user_id = user_id, language_id=sessions[(user_id, article_id)]['language']).first()

    

        timesTranslated = translated_words_per_article(user_id, article_id)
        userDurationWithTranslated = sessions[session]['user_duration'] - (timesTranslated * 3)
        sessions[session]['user_duration'] = userDurationWithTranslated
        
        if userDurationWithTranslated <= should_spend_reading_upper_bound and userDurationWithTranslated >= should_spend_reading_lower_bound and sessions[session]['liked'] == 0:
            have_read_sessions += 1
            sessions[session]['liked'] = 1
        elif sessions[session]['liked'] == 1:
            have_read_sessions += 1
        else:
            continue

    #df = pd.DataFrame.from_dict(sessions, orient='index')

    return sessions, have_read_sessions

#print(get_all_user_reading_sessions())

def get_df_and_plot_user_reading_sessions():
    sessions, have_read_sessions = get_all_user_reading_sessions()
    df = pd.DataFrame.from_dict(sessions, orient='index')
    plot_urs_with_duration_and_word_count(df, have_read_sessions, "test")
    return df

# This function makes a dataframe for one user with 
def isArticleLiked():
    user = 534
    
    articleData = {}

    # Not using this, new information found, average reading time for learners is lower.
    """ language_data = {
        2: {'name': 'Danish', 'avrt': 204}, # taken from norwegian
        3: {'name': 'German', 'avrt': 179},
        5: {'name': 'English', 'avrt': 228},
        7: {'name': 'French', 'avrt': 195},
        6: {'name': 'Spanish', 'avrt': 218},
        8: {'name': 'Italian', 'avrt': 188},
        9: {'name': 'Dutch', 'avrt': 202},
        10: {'name': 'Norwegian', 'avrt': 204},
        11: {'name': 'Portuguese', 'avrt': 181},
        13: {'name': 'Polish', 'avrt': 166},
        18: {'name': 'Swedish', 'avrt': 204}, # taken from norwegian
        19: {'name': 'Russian', 'avrt': 180}, # i made it
        23: {'name': 'Hungarian', 'avrt': 161}, # taken from finnish
    } """

    averageReadingTime = 70 # wpm
    
    readingSession = (
        UserReadingSession.query
            .filter_by(user_id=user)
            .filter(UserReadingSession.article_id.isnot(None))
            .filter(UserReadingSession.duration >= 30000) # 30 seconds
            .filter(UserReadingSession.duration <= 3600000) # 1 hour
            .order_by(UserReadingSession.article_id.asc())
            .all()
    )
    
    for session in readingSession:
        articleId = session.article_id
        article = Article.find_by_id(articleId)
        sessionDuration = int(session.duration) / 1000 # in seconds

        if articleId not in articleData:
            articleData[articleId] = {
                'user_duration': sessionDuration,
                'language': article.language_id,
                'difficulty': article.fk_difficulty,
                'word_count': article.word_count,
                'wpm': 0,
                #'expectedWpm': 0,
                'haveRead': 0,
                'days_since': (datetime.now() - session.start_time).days,
            }
        else:
            articleData[articleId]['user_duration'] += sessionDuration
    
    for article in articleData:
        # Not using this right now, but it could be useful
        # user_level = UserLanguage.query.filter_by(user_id = user, language_id=articleData[article]['language']).first()
        averageShouldSpendReading = get_expected_reading_time(articleData[article]['word_count'], 0)

        timesTranslated = translated_words_per_article(user, article)
        userDurationWithTranslated = articleData[article]['user_duration'] - (timesTranslated * 3)
 
        userReadWpm = articleData[article]['word_count'] / (userDurationWithTranslated / 60)

        articleData[article]['wpm'] = userReadWpm
        articleData[article]['expectedWpm'] = averageShouldSpendReading
        
        if (70 - 20) <= userReadWpm and userReadWpm <= (70 + 20):
            articleData[article]['haveRead'] = 1
        else:
            articleData[article]['haveRead'] = 0

        # not using atm
        """ articleData[article]['likeScore'] = int( 
            (articleData[article]['duration'] / averageShouldSpend ) 
            * (articleData[article]['difficulty'] / user_level.cefr_level)
        ) """

    df = pd.DataFrame.from_dict(articleData, orient='index')
    return df

print("before the function")

get_df_and_plot_user_reading_sessions()
#plot_urs_with_duration_and_word_count(isArticleLiked(), "oscar_test")

print("after test")