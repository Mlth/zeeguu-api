import sys

from elasticsearch import Elasticsearch
from zeeguu.core.elastic.settings import ES_CONN_STRING, ES_ZINDEX
from zeeguu.core.model import UserExerciseSession, User, UserReadingSession, Article, UserLanguage, UserActivityData, UserArticle
import pandas as pd
from zeeguu.core.model import db
import pyarrow as pa # needed for pandas

from zeeguu.api.app import create_app
from zeeguu.core.model.user_reading_session import UserReadingSession
from zeeguu.recommender.feedback_matrix import FeedbackMatrix

app = create_app()
app.app_context().push()

print("Running playground")

print("before the function")

matrix = FeedbackMatrix()
matrix.generate_dfs(True)
#matrix.plot_sessions_df("sessions")


# These are to just make a simple Dataframe that can be used for testing difficulty
#matrix.generate_simple_df()
matrix.plot_difficulty_sessions_df("difficulty_sessions_adjustment")
#matrix.plot_difficulty_sessions_df("difficulty_sessions_with_adjustment")

matrix.build_sparse_tensor()
matrix.visualize_tensor()

print("after test")