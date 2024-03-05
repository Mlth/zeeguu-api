import sys
import time
from elasticsearch import Elasticsearch
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage, UserActivityData, UserArticle
import pandas as pd
from zeeguu.core.model import db
import pyarrow as pa # needed for pandas
from zeeguu.api.app import create_app
from zeeguu.core.candidate_pool_generator.candidate_generator import build_candidate_pool_for_lang, build_candidate_pool_for_user
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig, ShowData
from zeeguu.core.elastic.elastic_query_builder import build_elastic_search_query as ElasticQuery
from zeeguu.core.elastic.indexing import index_all_articles

app = create_app()
app.app_context().push()

'''
print("before the for")
for id in User.all_recent_user_ids(150):
    u = User.find_by_id(id)
    print(u.name)
    duration_old = exercises_duration_by_day(u)
    duration_new = exercises_duration_by_day(u)
    if duration_new != duration_old:
        print("old way")
        print(duration_old)
        print("new way")
        print(duration_new)

=======
print("Running playground")

print("before the function")

print("before test")
u = User.find_by_id(534)
print(u.name)
lan = u.learned_language
'''
'''
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
'''

#res = build_candidate_pool_for_lang("en", "trump")
#print(len(res))

#for r in res:
#    print(r)

res = build_candidate_pool_for_user(534)

def articles_from_candidates(candidates):
    article_ids = [c.article_id for c in candidates]
    articles = Article.query.filter_by(broken=0).filter(Article.id.in_(article_ids)).all()
    return articles

def articles_from_candidates_2(candidates):
    article_ids = {c.article_id for c in candidates}
    max_id = candidates[len(candidates)-1].article_id
    min_id = candidates[0].article_id

    print("max", max_id)
    print("min", min_id)

    articles = Article.query.filter_by(broken=0).filter(Article.id >= min_id).filter(Article.id <= max_id).all()

    print(len(articles))

    filtered_articles = [a for a in articles if a.id in article_ids]

    return filtered_articles

start = time.time()

articles = articles_from_candidates_2(res)

end = time.time()
print(end - start)
print(articles[0])

print("query time:", len(res))

'''
#es2 = article_recommendations_for_user(u, 10)
term = "Ukraine"

res = article_search_for_user(u,20,term)
print(f"\nSearching for {term}\n")
for r in res: 
    print(r.topics)

print("\nSearching for standard recommendations\n The ones you see on the zeeguu main page\n")
res2 = article_recommendations_for_user(u,4)

for r in res2:
    print(r)

#print(res2)


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



#conn.close()

'''
'''
def initialize_all_focused_durations():
    for session in db['user_reading_session'].all():

        

def initialize_focused_duration(user_id, article_id):
    return
'''
print("Ending playground")
