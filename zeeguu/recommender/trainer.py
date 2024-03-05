from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig
from zeeguu.recommender.utils import ShowData


class Trainer:
    matrix_config = FeedbackMatrixConfig(
        show_data=[ShowData.ALL],
        adjustment_config=AdjustmentConfig(
            difficulty_weight=1,
            translation_adjustment_value=4
        ),
    )
    matrix = FeedbackMatrix(matrix_config)

    