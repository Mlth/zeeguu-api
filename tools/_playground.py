import sys

from elasticsearch import Elasticsearch
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage, UserActivityData, UserArticle
import pandas as pd
from zeeguu.core.model import db
import pyarrow as pa # needed for pandas

from zeeguu.api.app import create_app
from zeeguu.core.model.user_reading_session import UserReadingSession
from zeeguu.recommender.feedback_matrix import AdjustmentConfig, FeedbackMatrix, FeedbackMatrixConfig, ShowData

app = create_app()
app.app_context().push()

print("Running playground")

print("before the function")

matrix = FeedbackMatrix()

for i in range(5):
    print("round", str(i))
    config = FeedbackMatrixConfig(
        ShowData.RATED_DIFFICULTY, 
        AdjustmentConfig(
            difficulty_weight=1,
            translation_adjustment_value=i,
        )
    )

    matrix.generate_dfs(config)

    matrix.plot_sessions_df("test/run-" + str(i))

    print("round", str(i), "done")

print("after test")