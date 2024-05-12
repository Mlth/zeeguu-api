import time
from zeeguu.core.elastic.indexing import index_all_articles
from zeeguu.core.model import db
from zeeguu.api.app import create_app
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.mapper import Mapper
from datetime import timedelta, datetime
from zeeguu.recommender.mock.generators_mock import setup_session_5_likes_range, setup_session_2_categories, setup_sessions_4_categories_with_noise
from zeeguu.recommender.recommender_system import RecommenderSystem
from zeeguu.recommender.utils.recommender_utils import ShowData
from zeeguu.recommender.utils.train_utils import remove_saved_embeddings_and_mappings
import pandas as pd

app = create_app()
app.app_context().push()
print("Starting playground")
sesh = db.session

test = False
fresh = False

print("setting up config")

if(fresh):
    remove_saved_embeddings_and_mappings()

#data_since = datetime.now() - timedelta(days=49)
data_since = datetime(2024, 3, 14)

#data_since = None

start = time.time()
mapper = Mapper(data_since=data_since)


num_users = mapper.num_users
num_items = mapper.num_articles
max_article_id = mapper.max_article_id

print("Time to set up mapper: ", time.time() - start)

start = time.time()
matrix_config = FeedbackMatrixConfig(
    show_data=[ShowData.LIKED],
    data_since=data_since,
    adjustment_config=AdjustmentConfig(
        difficulty_weight=1,
        translation_adjustment_value=1
    ),
    test_tensor=test
)

matrix = FeedbackMatrix(matrix_config, mapper, num_users=num_users, num_articles=num_items)
matrix.generate_dfs()
matrix.plot_sessions_df('liked_sessions_df')
print("Time to generate dfs: ", time.time() - start)
""" 
start = time.time()
#sessions_df = pd.concat([matrix.liked_sessions_df, matrix.negative_sampling_df], ignore_index=True)
sessions_df = matrix.liked_sessions_df

if test:
    recommender = RecommenderSystem(sessions_df, mapper=mapper,num_users=1000, num_items=1000, generator_function=setup_sessions_4_categories_with_noise)
else:
    recommender = RecommenderSystem(sessions=sessions_df, num_users=num_users, num_items=num_items, data_since=data_since, mapper=mapper, embedding_dim=10)
print("Time to set up recommender: ", time.time() - start)

start = time.time()
recommender.cf_model.train_model(num_iterations=30000, learning_rate=0.1)
print("Time to train model: ", time.time() - start)

start = time.time()

if(test):
    recommender.user_recommendations(user_id=1, language_id=1)
else:
    recommender.user_recommendations(user_id=4557, language_id=2,more_like_this=False)

print("Time to get recommendations: ", time.time() - start)

print("Ending playground") """