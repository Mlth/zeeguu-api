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
matrix.calc_dfs()
matrix.plot_sessions_df("sessions")

print("after test")