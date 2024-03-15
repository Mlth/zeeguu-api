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

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format
def mask(df, key, function):
  """Returns a filtered dataframe, by applying function to key"""
  return df[function(df[key])]

def flatten_cols(df):
  df.columns = [' '.join(col).strip() for col in df.columns.values]
  return df

pd.DataFrame.mask = mask
pd.DataFrame.flatten_cols = flatten_cols

app = create_app()
app.app_context().push()

print("Starting playground")
sesh = db.session
initial_candidate_pool()

# Only temp solution. Set this to True if you want to use a very small user- and article space and only 2 sessions.
test = True

start_time = time.time()

matrix_config = FeedbackMatrixConfig(
    show_data=[],
    data_since=accurate_duration_date,
    adjustment_config=AdjustmentConfig(
        difficulty_weight=2,
        translation_adjustment_value=1
    ),
    test_tensor=test
)

matrix = FeedbackMatrix(matrix_config)
matrix.generate_dfs()

sessions_df = matrix.liked_sessions_df
matrix.plot_sessions_df("difficulty-parameter")

print("setting up config")

start_time = time.time()
matrix_config = FeedbackMatrixConfig(
        show_data=[],
        data_since=accurate_duration_date,
        adjustment_config=AdjustmentConfig(
            difficulty_weight=5,
            translation_adjustment_value=1
        ),
        test_tensor=test
    )
matrix = FeedbackMatrix(matrix_config)
matrix.generate_dfs()
liked_sessions_df = matrix.liked_sessions_df

sessions_df = matrix.liked_sessions_df
print("--- %s seconds ---" % (time.time() - start_time))

if test:
    recommender = RecommenderSystem(sessions_df, 100, 100, test=True)
else:
    recommender = RecommenderSystem(sessions_df, matrix.max_user_id, matrix.max_article_id)

start_time = time.time()

recommender.build_model()

recommender.cf_model.train()

if(test):
   recommender.user_recommendations(50)
else:
  recommender.user_recommendations(4338)
   

#recommender.visualize_article_embeddings()

print("--- %s seconds ---" % (time.time() - start_time))


print("Ending playground")
