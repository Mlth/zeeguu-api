import time
from zeeguu.core.model import db
from zeeguu.api.app import create_app
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.utils import accurate_duration_date
from zeeguu.recommender.mock.generators_mock import setup_session_5_likes_range, setup_session_2_categories
from zeeguu.recommender.recommender_system import RecommenderSystem

app = create_app()
app.app_context().push()
print("Starting playground")
sesh = db.session

test = True #enable this if you want constructed examples for debugging the recommendersystem.

start_time = time.time()

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
print("--- %s seconds for feedbackmatrix ---" % (time.time() - start_time))


if test:
    recommender = RecommenderSystem(sessions_df, 10, 10, test=True, generator_function=setup_session_2_categories)
else:
    recommender = RecommenderSystem(sessions_df, matrix.max_user_id, matrix.max_article_id)

start_time = time.time()

recommender.build_model()

recommender.cf_model.train()

if(test):
   recommender.user_recommendations(0)
else:
  recommender.user_recommendations(4338)
   
print("--- %s seconds --- for training and recommending" % (time.time() - start_time))

#TODO FIX FOR TEST CASES ASWELL 
#recommender.visualize_article_embeddings()
print("--- %s seconds --- for visualizations" % (time.time() - start_time))


print("Ending playground")