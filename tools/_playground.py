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

import tensorflow as tf
from zeeguu.recommender.recommender_system import RecommenderSystem

from zeeguu.recommender.recommender_system import RecommenderSystem
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

app = create_app()
app.app_context().push()

print("Starting playground")

matrix_config = FeedbackMatrixConfig(
    show_data=[ShowData.ALL],
    adjustment_config=AdjustmentConfig(
        difficulty_weight=1,
        translation_adjustment_value=4
    ),
)
matrix = FeedbackMatrix(matrix_config)
matrix.build_sparse_tensor()

liked_sessions_df = matrix.liked_sessions_df

# Define embedding layers for users and items
num_users = matrix.num_of_users  # Example: total number of users
num_items = matrix.num_of_articles  # Example: total number of items

recommender = RecommenderSystem(num_users, num_items)

#recommender.build_model(liked_sessions_df)

#recommender.cf_model.train()

#new_user_embeddings = recommender.cf_model.embeddings.get("user_id")
#new_article_embeddings = recommender.cf_model.embeddings.get("article_id")


print("Ending playground")