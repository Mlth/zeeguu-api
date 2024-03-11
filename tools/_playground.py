import sys
import time
from elasticsearch import Elasticsearch
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage, UserActivityData, UserArticle
import pandas as pd
from zeeguu.core.model import db
import pyarrow as pa # needed for pandas
from zeeguu.api.app import create_app
from zeeguu.recommender.candidate_generator import build_candidate_pool_for_lang, build_candidate_pool_for_user, initial_candidate_pool
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig, ShowData
from zeeguu.core.elastic.elastic_query_builder import build_elastic_search_query as ElasticQuery
from zeeguu.core.elastic.indexing import index_all_articles
from datetime import datetime, timedelta
from zeeguu.recommender.utils import accurate_duration_date

import tensorflow as tf
from zeeguu.recommender.recommender_system import RecommenderSystem

from zeeguu.recommender.recommender_system import RecommenderSystem
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

app = create_app()
app.app_context().push()

print("Starting playground")
sesh = db.session
initial_candidate_pool()

# Only temp solution. Set this to True if you want to use a very small user- and article space and only 2 sessions.
test = True

for i in range(5):
    matrix_config = FeedbackMatrixConfig(
        show_data=[],
        data_since=accurate_duration_date,
        adjustment_config=AdjustmentConfig(
            difficulty_weight=i,
            translation_adjustment_value=1
        ),
        test_tensor=test
    )

    matrix = FeedbackMatrix(matrix_config)
    matrix.generate_dfs()

    matrix.plot_sessions_df("difficulty-parameter:{}".format(i))

'''
liked_sessions_df = matrix.liked_sessions_df

# Define embedding layers for users and items
num_users = matrix.num_of_users
num_items = matrix.num_of_articles

if test:
    recommender = RecommenderSystem(500, 500)
else:
    recommender = RecommenderSystem(num_users, num_items)

recommender.build_model(liked_sessions_df)

recommender.cf_model.train()
'''

print("Ending playground")
