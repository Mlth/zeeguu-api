from zeeguu.api.app import create_app
from .mapper import Mapper
from . import feedback_matrix

from datetime import timedelta, datetime
from .recommender_system import (
    RecommenderSystem,

)
class Singleton:
   
    

    def makeRecommender(self):
        app = create_app()
        app.app_context().push()
        test = False

        mapper = Mapper()
        
        num_users = mapper.num_users
        num_items = mapper.num_articles

        matrix_config = feedback_matrix.FeedbackMatrixConfig(
            show_data=[],
            data_since=datetime.now() - timedelta(days=130), # accurate_duration_date
            adjustment_config=feedback_matrix.AdjustmentConfig(
                difficulty_weight=2,
                translation_adjustment_value=1
            ),
            test_tensor=test
        )

        matrix = feedback_matrix.FeedbackMatrix(matrix_config, mapper, num_users=num_users, num_articles=num_items)

        matrix.generate_dfs()

        return RecommenderSystem(matrix.liked_sessions_df, mapper=mapper,num_users=num_users, num_items=num_items)