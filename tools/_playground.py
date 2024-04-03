import time
from elasticsearch import Elasticsearch
from zeeguu.core.model import db
import sqlalchemy as database
from zeeguu.api.app import create_app
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
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

matrix_config = FeedbackMatrixConfig(
    show_data=[],
    data_since=accurate_duration_date, #datetime.now() - timedelta(days=365), # accurate_duration_date
    adjustment_config=AdjustmentConfig(
        difficulty_weight=2,
        translation_adjustment_value=1
    ),
    test_tensor=test
)

matrix = FeedbackMatrix(matrix_config)
matrix.generate_dfs()

sessions_df = matrix.liked_sessions_df

if test:
    recommender = RecommenderSystem(sessions_df, 1000, 1000, generator_function=setup_sessions_4_categories_with_noise)
else:
    recommender = RecommenderSystem(sessions=sessions_df, num_users=matrix.max_user_id, num_items=matrix.max_article_id)

recommender.cf_model.train_model(num_iterations=40000, learning_rate=0.05)

if(test):
    recommender.user_recommendations(user_id=1)
else:
    recommender.user_recommendations(user_id=4338, language_id=9)

print("Ending playground")