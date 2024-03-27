import time
from elasticsearch import Elasticsearch
import zeeguu
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage, UserActivityData, UserArticle
import pandas as pd
from zeeguu.core.model import db
import sqlalchemy as database
from zeeguu.api.app import create_app
from zeeguu.recommender.candidate_generator import initial_candidate_pool
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.opti_feedback_matrix import OptiAdjustmentConfig, OptiFeedbackMatrix, OptiFeedbackMatrixConfig
from zeeguu.recommender.utils import accurate_duration_date, get_dataframe_user_reading_sessions, setup_df_correct, ShowData
from datetime import timedelta, datetime
from zeeguu.recommender.mock.generators_mock import setup_session_5_likes_range, setup_session_2_categories
from zeeguu.recommender.recommender_system import RecommenderSystem

app = create_app()
app.app_context().push()
print("Starting playground")
sesh = db.session
initial_candidate_pool()

# Only temp solution. Set this to True if you want to use a very small user- and article space and only 2 sessions.
test = True

start_time = time.time()
print("setting up config")

matrix_config = FeedbackMatrixConfig(
    show_data=[ShowData.LIKED, ShowData.RATED_DIFFICULTY],
    data_since= datetime.now() - timedelta(days=365), # accurate_duration_date
    adjustment_config=AdjustmentConfig(
        difficulty_weight=2,
        translation_adjustment_value=1
    ),
    test_tensor=test
)

matrix = FeedbackMatrix(matrix_config)
matrix.generate_dfs()

sessions_df = matrix.liked_sessions_df

#setup_df_correct(matrix.max_article_id)

""" print("Querying sessions df")
print(len(matrix.sessions_df))

print("Querying liked sessions df")
print(len(sessions_df))
print(sessions_df.columns)
print(sessions_df) """

#matrix.plot_sessions_df("difficulty-parameter")


print("--- %s seconds ---" % (time.time() - start_time))
""" DB_URI = zeeguu.core.app.config["SQLALCHEMY_DATABASE_URI"]
engine = database.create_engine(DB_URI)

start_time = time.time()

matrix_config = OptiFeedbackMatrixConfig(
    show_data=[ShowData.LIKED, ShowData.RATED_DIFFICULTY],
    data_since= datetime.now() - timedelta(days=365), # accurate_duration_date
    adjustment_config=OptiAdjustmentConfig(
        difficulty_weight=2,
        translation_adjustment_value=1
    ),
    test_tensor=test
)

opti_matrix = OptiFeedbackMatrix(matrix_config)
opti_matrix.generate_opti_dfs()

user_df = get_dataframe_user_reading_sessions(datetime.now() - timedelta(days=365))
print("User_df query ")
print(user_df.columns)
print(len(user_df))

sessions_df_from_opti = opti_matrix.sessions_df
print("Opti matrix query")
print(len(sessions_df_from_opti))

liked_sessions_df_from_opti = opti_matrix.liked_sessions_df
print("Liked sessions query")
print(len(liked_sessions_df_from_opti))
print(liked_sessions_df_from_opti.columns)

print("Opti matrix have_read_sessions")
print(opti_matrix.have_read_sessions) """





print("--- %s seconds ---" % (time.time() - start_time))

#path = "./zeeguu/recommender/embeddings/"
#recommender.save_embeddings(path)


if test:
    recommender = RecommenderSystem(sessions_df, 500, 500, test=True, generator_function=setup_session_2_categories)
else:
    recommender = RecommenderSystem(sessions_df, matrix.max_user_id, matrix.max_article_id)

recommender.build_regularized_model()

recommender.cf_model.train(num_iterations=50000, learning_rate=0.15)

if(test):
    recommender.user_recommendations(1)
else:
    recommender.user_recommendations(4338)

if test:
    user_liked_articles = list(recommender.sessions[recommender.sessions['user_id'] == 1]['article_id'])
    recommender.visualize_article_embeddings(marked_articles=user_liked_articles)

print("Ending playground")