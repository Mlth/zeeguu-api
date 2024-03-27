import time
from zeeguu.core.model import db
from zeeguu.api.app import create_app
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.utils import accurate_duration_date, get_resource_path
from zeeguu.recommender.mock.generators_mock import setup_session_5_likes_range, setup_session_2_categories, setup_sessions_4_categories_with_noise
from zeeguu.recommender.recommender_system import Measure, RecommenderSystem
import pandas as pd
from datetime import datetime

app = create_app()
app.app_context().push()
print("Starting playground")

test = False #enable this if you want constructed examples for debugging the recommendersystem.

matrix_config = FeedbackMatrixConfig(
        show_data=[],
        data_since=datetime.now() - datetime.timedelta(days=365),
        adjustment_config=AdjustmentConfig(
            difficulty_weight=5,
            translation_adjustment_value=1
        ),
        test_tensor=test
    )

matrix = FeedbackMatrix(matrix_config)
matrix.generate_dfs()

sessions_df = matrix.liked_sessions_df

print(len(sessions_df))

'''if test:
    recommender = RecommenderSystem(sessions_df, 500, 500, test=True, generator_function=setup_sessions_4_categories_with_noise)
else:
    recommender = RecommenderSystem(sessions_df, matrix.max_user_id, matrix.max_article_id)

recommender.build_regularized_model()

recommender.cf_model.train(num_iterations=500, learning_rate=0.25)


if(test):
    recommender.user_recommendations(1)
else:
    recommender.user_recommendations(4338)'''


#recommender.sessions.to_csv(get_resource_path() + 'sessions.csv', index=False)
#recommender.article_neighbors(article_id=30)

#if test:
#    user_liked_articles = list(recommender.sessions[recommender.sessions['user_id'] == 1]['article_id'])
#    recommender.visualize_article_embeddings(marked_articles=user_liked_articles)

print("Ending playground")