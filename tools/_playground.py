import time
from elasticsearch import Elasticsearch
from zeeguu.core.model import db
import sqlalchemy as database
from zeeguu.api.app import create_app
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.mapper import Mapper
from zeeguu.recommender.opti_feedback_matrix import OptiAdjustmentConfig, OptiFeedbackMatrix, OptiFeedbackMatrixConfig
from datetime import timedelta, datetime
from zeeguu.recommender.mock.generators_mock import setup_session_5_likes_range, setup_session_2_categories, setup_sessions_4_categories_with_noise
from zeeguu.recommender.recommender_system import RecommenderSystem

app = create_app()
app.app_context().push()
print("Starting playground")
sesh = db.session

# Only temp solution. Set this to True if you want to use a very small user- and article space and only 2 sessions.
test = False

print("setting up config")


start = time.time()
mapper = Mapper()

num_users = mapper.num_users
num_items = mapper.num_articles
print("Time to set up mapper: ", time.time() - start)


start = time.time()
matrix_config = FeedbackMatrixConfig(
    show_data=[],
    data_since=datetime.now() - timedelta(days=130), # accurate_duration_date
    adjustment_config=AdjustmentConfig(
        difficulty_weight=2,
        translation_adjustment_value=1
    ),
    test_tensor=test
)

matrix = FeedbackMatrix(matrix_config, mapper, num_users=num_users, num_articles=num_items)
matrix.generate_dfs()
print("Time to generate dfs: ", time.time() - start)


start = time.time()
sessions_df = matrix.liked_sessions_df

if test:
    recommender = RecommenderSystem(sessions_df, mapper=mapper,num_users=1000, num_items=1000, generator_function=setup_sessions_4_categories_with_noise)
else:
    recommender = RecommenderSystem(sessions=sessions_df, num_users=num_users, num_items=num_items, mapper=mapper)
print("Time to set up recommender: ", time.time() - start)


start = time.time()
recommender.cf_model.train_model(num_iterations=5000, learning_rate=0.05)
print("Time to train model: ", time.time() - start)


start = time.time()

if(test):
    recommender.user_recommendations(user_id=1, language_id=1)
else:
    recommender.user_recommendations(user_id=535, language_id=9)

print("Time to get recommendations: ", time.time() - start)

print("Ending playground")