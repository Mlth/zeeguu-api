import time
from zeeguu.core.model import db
from zeeguu.api.app import create_app
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.mapper import Mapper
from datetime import timedelta, datetime
from zeeguu.recommender.mock.generators_mock import setup_session_5_likes_range, setup_session_2_categories, setup_sessions_4_categories_with_noise
from zeeguu.recommender.recommender_system import RecommenderSystem
from zeeguu.recommender.utils.train_utils import remove_saved_embeddings_and_mappings

app = create_app()
app.app_context().push()
print("Starting playground")
sesh = db.session

test = False
fresh = False

print("setting up config")


if(fresh):
    remove_saved_embeddings_and_mappings()

start = time.time()
mapper = Mapper()

num_users = mapper.num_users
num_items = mapper.num_articles
print("Time to set up mapper: ", time.time() - start)



start = time.time()
matrix_config = FeedbackMatrixConfig(
    show_data=[], 
    #data_since = timedelta(days=30),
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
recommender.cf_model.train_model(num_iterations=30000, learning_rate=0.1)
print("Time to train model: ", time.time() - start)


start = time.time()

if(test):
    recommender.user_recommendations(user_id=1, language_id=1)
else:
    recommender.user_recommendations(user_id=535, language_id=9)

print("Time to get recommendations: ", time.time() - start)

print("Ending playground")