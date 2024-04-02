import time
from elasticsearch import Elasticsearch
from zeeguu.core.model import db
import sqlalchemy as database
from zeeguu.api.app import create_app
from zeeguu.recommender.candidate_generator import initial_candidate_pool
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.opti_feedback_matrix import OptiAdjustmentConfig, OptiFeedbackMatrix, OptiFeedbackMatrixConfig
from zeeguu.recommender.utils import accurate_duration_date, get_dataframe_user_reading_sessions, setup_df_correct, ShowData
from datetime import timedelta, datetime
from zeeguu.recommender.mock.generators_mock import setup_session_5_likes_range, setup_session_2_categories, setup_sessions_4_categories_with_noise
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

start_time = time.time()
print("SETTING UP RECOMMENDING")

if test:
    recommender = RecommenderSystem(sessions_df, 500, 500, test=test, generator_function=setup_sessions_4_categories_with_noise)
else:
    recommender = RecommenderSystem(sessions_df, matrix.max_user_id, matrix.max_article_id)

recommender.train_model(num_iterations=50000, learning_rate=0.15)

if(test):
    recommender.user_recommendations(1)
else:
    recommender.user_recommendations(4338)

print("--- %s seconds --- FOR RECOMMEDING" % (time.time() - start_time))

print("Ending playground")