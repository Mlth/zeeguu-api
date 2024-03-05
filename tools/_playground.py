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

print("Starting playground")

'''
u = User.find_by_id(534)
print(u.name)
lan = u.learned_language

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


'''def articles_from_candidates_2(candidates):
    article_ids = {c.article_id for c in candidates}
    max_id = candidates[len(candidates)-1].article_id
    min_id = candidates[0].article_id

    print("max", max_id)
    print("min", min_id)

    articles = Article.query.filter_by(broken=0).filter(Article.id >= min_id).filter(Article.id <= max_id).all()

    print(len(articles))

    filtered_articles = [a for a in articles if a.id in article_ids]

    return filtered_articles'''


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
        [ShowData.ALL, ShowData.NEW_DATA], 
        AdjustmentConfig(
            difficulty_weight=i,
            translation_adjustment_value=1,
        )
    )

    matrix = FeedbackMatrix(config)

    matrix.generate_dfs()

    matrix.plot_sessions_df("test/with_difficulty-" + str(i))

    print("round", str(i), "done")

print("Ending playground")
