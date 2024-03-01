import sys

from elasticsearch import Elasticsearch
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage, UserActivityData, UserArticle
import pandas as pd
from zeeguu.core.model import db
import pyarrow as pa # needed for pandas

from zeeguu.api.app import create_app
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig, ShowData
from zeeguu.core.elastic.elastic_query_builder import build_elastic_search_query as ElasticQuery
from zeeguu.core.elastic.indexing import index_all_articles

app = create_app()
app.app_context().push()

print("Running playground")

print("before the function")

print("before test")
u = User.find_by_id(1)
print(u.name)
lan = u.learned_language

res = ElasticQuery(
    10,
    "cake",
    "sports, politics",
    "",
    "",
    "",
    lan,
    10,
    1
)
print (res)


index_all_articles(db.session)

#es2 = article_recommendations_for_user(u, 10)

#print(res2)


'''
conn = db.engine.raw_connection()

for i in range(5):
    print("round", str(i))
    config = FeedbackMatrixConfig(
        ShowData.RATED_DIFFICULTY, 
        AdjustmentConfig(
            difficulty_weight=1,
            translation_adjustment_value=i,
        )
    )

    matrix.generate_dfs(config)

    matrix.plot_sessions_df("test/run-" + str(i))

    print("round", str(i), "done")

df = pd.read_sql_query(query, conn)
#df.to_csv(sys.stdout, index=False)
df.astype('int32').dtypes
#df.plot(kind = 'scatter', x = 'duration', y = 'word_count', color='blue')

#if upper_bound:
#    x_values = df['duration']
#    y_values_line = 20 * x_values + y_start
#    plt.scatter(df['duration'], df['word_count'], label='Data Points')
#    plt.plot(x_values, y_values_line, color='red', label='y = 2x + 2')
if lower_bound:
    x_values = df['duration']
    y_values_line = [y_start] * len(x_values)
    plt.scatter(df['duration'], df['word_count'], label='Data Points')
    plt.plot(x_values, y_values_line, color='red', label='y = 2x + 2')
'''


#conn.close()

'''
def initialize_all_focused_durations():
    for session in db['user_reading_session'].all():

        

def initialize_focused_duration(user_id, article_id):
    return
'''
