import sys

from elasticsearch import Elasticsearch
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage
import pandas as pd
from zeeguu.core.model import db
import matplotlib.pyplot as plt
#import numpy as np

from zeeguu.api.app import create_app
from zeeguu.core.model.user_reading_session import UserReadingSession

app = create_app()
app.app_context().push()

# This function makes a dataframe for one user with 
def isArticleLiked():
    user = 534
    

    articleData = {}

    language_data = {
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
    }

 
    readingSession = ( 
        UserReadingSession.query
            .filter_by(user_id=user)
            .filter(UserReadingSession.article_id.isnot(None))
            .order_by(UserReadingSession.article_id.asc())
            .limit(100)
    )
    for session in readingSession:
        articleId = session.article_id
        article = Article.find_by_id(articleId)
        sessionDuration = int(session.duration) / 1000
        if articleId not in articleData:
            articleData[articleId] = {
                'duration': sessionDuration,
                'language': article.language_id,
                'difficulty': article.fk_difficulty,
                'wordCount': article.word_count,
                'likeScore': 0
            }
        else:
            articleData[articleId]['duration'] += sessionDuration

    
    for article in articleData:
        user_level = UserLanguage.query.filter_by(user_id = user, language_id=articleData[article]['language']).first()
        averageShouldSpend = articleData[article]['wordCount'] / language_data[articleData[article]['language']]['avrt']
       

        articleData[article]['likeScore'] = int( 
            (articleData[article]['duration'] / averageShouldSpend ) 
            * (articleData[article]['difficulty'] / user_level.cefr_level)
        )

    df = pd.DataFrame.from_dict(articleData, orient='index')
    return df

print("before the function")
print(isArticleLiked())

# print("before the for")
""" for id in User.all_recent_user_ids(150):
    u = User.find_by_id(id)
    print(u.name)
    duration_old = exercises_duration_by_day(u)
    duration_new = exercises_duration_by_day(u)
    if duration_new != duration_old:
        print("old way")
        print(duration_old)
        print("new way")
        print(duration_new) """


# print("before test")

conn = db.engine.raw_connection()

query = """
    SELECT * from user_reading_session urs
    LEFT JOIN article a ON urs.article_id = a.id
    WHERE urs.duration / a.word_count > 0.1
"""

query = """
    SELECT
        (urs.duration / 60 / 60) AS duration,
        a.word_count
    FROM
        user_reading_session urs
    LEFT JOIN
        article a ON urs.article_id = a.id
    JOIN
        (SELECT article_id, COUNT(*) AS session_count
        FROM user_reading_session
        GROUP BY article_id
        HAVING COUNT(*) >= 20) AS session_counts
    ON
        urs.article_id = session_counts.article_id
    WHERE
        urs.duration IS NOT NULL
        AND a.word_count IS NOT NULL
        AND urs.duration / 60 / 60 < 60
        AND urs.duration / 60 / 60 >= 2
        AND a.word_count < 2000
"""

upper_bound = True
lower_bound = True

y_start = 50

df = pd.read_sql_query(query, conn)
#df.to_csv(sys.stdout, index=False)
df.astype('int32').dtypes
df.plot(kind = 'scatter', x = 'duration', y = 'word_count', color='blue')

if upper_bound:
    x_values = df['duration']
    y_values_line = 20 * x_values + y_start
    plt.scatter(df['duration'], df['word_count'], label='Data Points')
    plt.plot(x_values, y_values_line, color='red', label='y = 2x + 2')

if lower_bound:
    x_values = df['duration']
    y_values_line = [y_start] * len(x_values)
    plt.scatter(df['duration'], df['word_count'], label='Data Points')
    plt.plot(x_values, y_values_line, color='red', label='y = 2x + 2')

plt.savefig('test.png')
print("Has been saved")
plt.show()

conn.close()

print("after test") 

'''
def initialize_all_focused_durations():
    for session in db['user_reading_session'].all():

        

def initialize_focused_duration(user_id, article_id):
    return
'''