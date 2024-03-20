import time
from zeeguu.core.model import db
from zeeguu.api.app import create_app
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.utils import accurate_duration_date
from zeeguu.recommender.visualization.tensor_visualizer import setup_session_5_likes_range
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

generator_function = setup_session_5_likes_range

if test:
    recommender = RecommenderSystem(sessions_df, 100, 100, test=True, generator_function=generator_function)
else:
    recommender = RecommenderSystem(sessions_df, matrix.max_user_id, matrix.max_article_id)

start_time = time.time()

recommender.build_model()

recommender.cf_model.train()

if(test):
   recommender.user_recommendations(2)
else:
  recommender.user_recommendations(4338)
   
print("--- %s seconds --- for training and recommending" % (time.time() - start_time))


#this takes a very long time.. hmm
recommender.visualize_article_embeddings()
print("--- %s seconds --- for visualizations" % (time.time() - start_time))


print("Ending playground")