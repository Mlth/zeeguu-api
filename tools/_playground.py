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

test = True #enable this if you want constructed examples for debugging the recommendersystem.

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

sessions_df = matrix.liked_sessions_df

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