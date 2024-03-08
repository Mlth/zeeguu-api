from typing import Tuple, List
from zeeguu.core.model.article import Article
from zeeguu.core.model.article_difficulty_feedback import ArticleDifficultyFeedback
from zeeguu.core.model.user import User
from zeeguu.core.model.user_activitiy_data import UserActivityData
from zeeguu.core.model.user_article import UserArticle
from zeeguu.core.model.user_language import UserLanguage
from zeeguu.core.model.user_reading_session import UserReadingSession
from zeeguu.recommender.tensor_utils import build_liked_sparse_tensor
from zeeguu.recommender.utils import cefr_to_fk_difficulty, get_diff_in_article_and_user_level, get_expected_reading_time, lower_bound_reading_speed, upper_bound_reading_speed, ShowData
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from zeeguu.core.model import db
from collections import Counter
from zeeguu.recommender.visualizer import Visualizer
from zeeguu.core.model import db
from sqlalchemy import or_, and_

import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class FeedbackMatrixSession:
    def __init__(self, user_id, article_id, session_duration, language_id, difficulty, word_count, article_topic_list, expected_read, liked, difficulty_feedback, days_since):
        self.user_id = user_id
        self.article_id = article_id
        self.session_duration = session_duration
        self.original_session_duration = session_duration
        self.language_id = language_id
        self.difficulty = difficulty
        self.word_count = word_count
        self.article_topic_list = article_topic_list
        self.expected_read = expected_read
        self.original_expected_read = expected_read
        self.liked = liked
        self.difficulty_feedback = difficulty_feedback
        self.days_since = days_since

class AdjustmentConfig:
    def __init__(self, difficulty_weight, translation_adjustment_value):
        self.difficulty_weight = difficulty_weight
        self.translation_adjustment_value = translation_adjustment_value

class FeedbackMatrixConfig:
    def __init__(self, show_data: List[ShowData], adjustment_config: AdjustmentConfig):
        self.show_data = show_data
        self.adjustment_config = adjustment_config

class FeedbackMatrix:
    default_difficulty_weight = 1
    default_translation_adjustment_value = 3

    tensor = None
    sessions_df = None
    liked_sessions_df = None
    have_read_sessions = None
    feedback_diff_list_toprint = None
    feedback_counter = 0

    num_of_users = None
    num_of_articles = None

    article_order_to_id = {}
    article_id_to_order = {}
    user_order_to_id = {}
    user_id_to_order = {}

    visualizer = Visualizer()

    def __init__(self, config: FeedbackMatrixConfig):
        self.config = config

    def get_user_reading_sessions(self, show_data: List[ShowData] = ShowData.ALL):
        print("Getting all user reading sessions")
        query = (
            UserReadingSession.query
                .join(User, User.id == UserReadingSession.user_id)
                .join(Article, Article.id == UserReadingSession.article_id)
                .filter(Article.broken == 0)
                .filter(User.is_dev == False)
                .filter(UserReadingSession.article_id.isnot(None))
                .filter(UserReadingSession.duration >= 30000) # 30 seconds
                .filter(UserReadingSession.duration <= 3600000) # 1 hour
                .filter(UserReadingSession.start_time >= datetime.now() - timedelta(days=365)) # 1 year
                .order_by(UserReadingSession.user_id.asc())
        )
        or_filters = []
        and_filters = []
        if ShowData.LIKED in show_data:
            query = (
                query.join(UserArticle, (UserArticle.article_id == UserReadingSession.article_id) & (UserArticle.user_id == UserReadingSession.user_id), isouter=True)
            )
            or_filters.append(UserArticle.liked == True)
        if ShowData.RATED_DIFFICULTY in show_data:
            query = (
                query.join(ArticleDifficultyFeedback, (ArticleDifficultyFeedback.article_id == UserReadingSession.article_id) & (ArticleDifficultyFeedback.user_id == UserReadingSession.user_id), isouter=True)
            )
            or_filters.append(ArticleDifficultyFeedback.difficulty_feedback.isnot(None))
        if ShowData.NEW_DATA in show_data:
            and_filters.append(UserReadingSession.start_time >= datetime(day=30, month=1, year=2024))
        if len(or_filters) > 0:
            query = query.filter(or_(*or_filters))
        if len(and_filters) > 0:
            query = query.filter(and_(*and_filters))
        return query.all()

    def get_sessions(self):
        print("Getting sessions")
        sessions: dict[Tuple[int, int], FeedbackMatrixSession] = {}

        query_data = None
        query_data = self.get_user_reading_sessions(self.config.show_data)

        for session in query_data:
            article_id = session.article_id
            user_id = session.user_id
            article = Article.find_by_id(article_id)
            session_duration = int(session.duration) / 1000 # in seconds
            liked = UserArticle.query.filter_by(user_id=user_id, article_id=article_id).with_entities(UserArticle.liked).first()
            liked_value = 0 if liked == (False,) or liked is None else 1 # should check out
            difficulty_feedback = ArticleDifficultyFeedback.query.filter_by(user_id=user_id, article_id=article_id).with_entities(ArticleDifficultyFeedback.difficulty_feedback).first()
            difficulty_feedback_value = 0 if difficulty_feedback is None else int(difficulty_feedback[0])
            
            if difficulty_feedback_value != 0:
                self.feedback_counter += 1

            article_topic = article.topics
            article_topic_list = []
            if len(article_topic) > 0:
                for topic in article_topic:
                    article_topic_list.append(topic.title)

            if (user_id, article_id) not in sessions:
                sessions[(user_id, article_id)] = FeedbackMatrixSession(
                    user_id,
                    article_id,
                    session_duration,
                    article.language_id,
                    article.fk_difficulty,
                    article.word_count,
                    article_topic_list,
                    0,
                    liked_value,
                    difficulty_feedback_value,
                    (datetime.now() - session.start_time).days,
                )
            else:
                sessions[(user_id, article_id)].session_duration += session_duration

        return self.get_sessions_data(sessions)
    
    def get_sessions_data(self, sessions: dict[Tuple[int, int], FeedbackMatrixSession]):
        liked_sessions = []
        feedback_diff_list = []
        have_read_sessions = 0

        if self.config.adjustment_config is None:
            self.config.adjustment_config = AdjustmentConfig(difficulty_weight=self.default_difficulty_weight, translation_adjustment_value=self.default_translation_adjustment_value)

        for session in sessions.keys():
            sessions[session].session_duration = self.get_translation_adjustment(sessions[session], self.config.adjustment_config.translation_adjustment_value)
            sessions[session].session_duration = self.get_difficulty_adjustment(sessions[session], self.config.adjustment_config.difficulty_weight)

            should_spend_reading_lower_bound = get_expected_reading_time(sessions[session].word_count, upper_bound_reading_speed)
            should_spend_reading_upper_bound = get_expected_reading_time(sessions[session].word_count, lower_bound_reading_speed)

            if self.duration_is_within_bounds(sessions[session].session_duration, should_spend_reading_lower_bound, should_spend_reading_upper_bound):
                have_read_sessions += 1
                sessions[session].expected_read = 1
                liked_sessions.append(sessions[session])
                feedback_diff_list.append(sessions[session].difficulty_feedback)
            if self.duration_is_within_bounds(sessions[session].original_session_duration, should_spend_reading_lower_bound, should_spend_reading_upper_bound):
                sessions[session].original_expected_read = 1
        
        return sessions, liked_sessions, have_read_sessions, feedback_diff_list

    def get_difficulty_adjustment(self, session: FeedbackMatrixSession, weight):
        user_level_query = (
            UserLanguage.query
                .filter_by(user_id = session.user_id, language_id=session.language_id)
                .filter(UserLanguage.cefr_level.isnot(None))
                .with_entities(UserLanguage.cefr_level)
                .first()
        )
        
        if user_level_query is None or user_level_query[0] == 0 or user_level_query[0] is None or user_level_query[0] == [] or user_level_query == []:
            return session.session_duration
        user_level = user_level_query[0]

        difficulty = session.difficulty
        fk_difficulty = cefr_to_fk_difficulty(difficulty)
        return session.session_duration * get_diff_in_article_and_user_level(fk_difficulty, user_level, weight)

    def get_translation_adjustment(self, session: FeedbackMatrixSession, adjustment_value):
        timesTranslated = UserActivityData.translated_words_for_article(session.user_id, session.article_id)
        return session.session_duration - (timesTranslated * adjustment_value)

    def duration_is_within_bounds(self, duration, lower, upper):
        return duration <= upper and duration >= lower

    def set_article_order_to_id(self):
        articles = Article.query.filter(Article.broken == 0).all()
        index = 0
        for article in articles:
            self.article_order_to_id[index] = article.id
            self.article_id_to_order[article.id] = index
            index += 1

    def set_user_order_to_id(self):
        users = User.query.filter(User.is_dev == False).all()
        index = 0
        for user in users:
            self.user_order_to_id[index] = user.id
            self.user_id_to_order[user.id] = index
            index += 1

    def generate_dfs(self):
        self.set_article_order_to_id()
        self.set_user_order_to_id()

        sessions, liked_sessions, have_read_sessions, feedback_diff_list = self.get_sessions()

        for i in range(len(liked_sessions)):
            liked_sessions[i].user_id = self.user_id_to_order.get(liked_sessions[i].user_id)
            liked_sessions[i].article_id = self.article_id_to_order.get(liked_sessions[i].article_id)

        df = self.__session_map_to_df(sessions)
        liked_df = self.__session_list_to_df(liked_sessions)
        #liked_df = self.__session_list_to_df([FeedbackMatrixSession(1, 1, 1, 1, 1, 1, [1], 1, 1, 1, 1), FeedbackMatrixSession(505, 510, 100, 5, 5, 100, [1], 1, 1, 1, 20)])

        self.sessions_df = df
        self.liked_sessions_df = liked_df
        self.have_read_sessions = have_read_sessions
        self.feedback_diff_list_toprint = feedback_diff_list

    def __session_map_to_df(self, sessions: dict[Tuple[int, int], FeedbackMatrixSession]):
        data = {index: vars(session) for index, session in sessions.items()}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df

    def __session_list_to_df(self, sessions: list[FeedbackMatrixSession]):
        # Pretty weird logic. We convert a list to a dict and then to a dataframe. Should be changed.
        data = {index: vars(session) for index, session in enumerate(sessions)}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df

    def build_sparse_tensor(self, force=False):
        # This function is not run in the constructor because it takes such a long time to run.
        print("Building sparse tensor")
        if (self.liked_sessions_df is None or self.sessions_df is None or self.have_read_sessions is None) or force:
            self.generate_dfs()

        self.num_of_users = User.num_of_users()
        self.num_of_articles = Article.num_of_articles()

        self.tensor = build_liked_sparse_tensor(self.liked_sessions_df, self.num_of_users, self.num_of_articles)

    def plot_sessions_df(self, name):
        print("Plotting sessions. Saving to file: " + name + ".png")
        self.visualizer.plot_urs_with_duration_and_word_count(self.sessions_df, self.have_read_sessions, name, self.config.show_data)

    def visualize_tensor(self, file_name='tensor'):
        print("Visualizing tensor")

        if self.tensor is None:
            print("Tensor is None. Building tensor first")
            self.build_sparse_tensor()

        self.visualizer.visualize_tensor(self.tensor, file_name)

    def print_feedback_difficulty_list(self, name):
        element_counts = Counter(self.feedback_diff_list_toprint)

        with open(name + ".txt", 'w') as f:
            f.write(f"Total number of feedback recorded from the session: {self.feedback_counter}\n")
            f.write(f"The amount of different values inside the proposed range:\n")
            for element, count in element_counts.items():
                f.write(f"Difficulty: {element}: count: {count}\n")
            f.write(f"The number of feedbacks insde the range: {len(self.feedback_diff_list_toprint)}\n")    
            f.write(f"The number of feedbacks outside the range: {self.feedback_counter - len(self.feedback_diff_list_toprint)}\n")    