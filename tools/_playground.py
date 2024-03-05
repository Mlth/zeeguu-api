import sys
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
