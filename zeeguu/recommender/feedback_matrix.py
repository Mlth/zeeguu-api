from zeeguu.core.model.article import Article
from zeeguu.core.model.article_difficulty_feedback import ArticleDifficultyFeedback
from zeeguu.core.model.user import User
from zeeguu.core.model.user_activitiy_data import UserActivityData
from zeeguu.core.model.user_article import UserArticle
from zeeguu.recommender.tensor_utils import build_liked_sparse_tensor
from zeeguu.recommender.utils import get_expected_reading_time, lower_bound_reading_speed, upper_bound_reading_speed, ShowData, get_difficulty_adjustment, get_user_reading_sessions
from datetime import datetime
import pandas as pd
from collections import Counter
from zeeguu.recommender.visualization.session_visualizer import SessionVisualizer
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
    '''Adjustments made to a sessions depending on 
    1. Variance in user fk and article fk
    2. Number of translated words in the article'''
    def __init__(self, difficulty_weight : int, translation_adjustment_value: int):
        self.difficulty_weight = difficulty_weight
        self.translation_adjustment_value = translation_adjustment_value

class FeedbackMatrixConfig:
    def __init__(self, adjustment_config: AdjustmentConfig, show_data: list[ShowData] =[], test_tensor: bool = False, data_since: datetime = None):
        self.adjustment_config = adjustment_config
        self.show_data = show_data
        self.data_since = data_since
        self.test_tensor = test_tensor

class FeedbackMatrix:
    default_difficulty_weight = 1
    default_translation_adjustment_value = 3
    
    tensor = None
    sessions_df = None
    liked_sessions_df = None
    have_read_sessions = None
    feedback_diff_list_toprint = None
    feedback_counter = 0

    def __init__(self, config: FeedbackMatrixConfig):
        self.config = config
        self.num_of_users = User.num_of_users()
        self.num_of_articles = Article.num_of_articles()
        self.visualizer = SessionVisualizer()
        self.max_article_id = Article.query.filter(Article.broken == 0).order_by(Article.id.desc()).first().id
        self.max_user_id = User.query.filter(User.is_dev == False).order_by(User.id.desc()).first().id

    def get_sessions(self):
        '''Gets all user reading sessions with respect to the given config'''
        print("Getting sessions")
        sessions: dict[tuple[int, int], FeedbackMatrixSession] = {}
        query_data = get_user_reading_sessions(self.config.data_since, self.config.show_data)

        for session in query_data:
            article_id = session.article_id
            user_id = session.user_id
            article = Article.find_by_id(article_id)
            session_duration = int(session.duration) / 1000 # in seconds
            liked = UserArticle.query.filter_by(user_id=user_id, article_id=article_id).with_entities(UserArticle.liked).first()
            if liked == (False,) or liked is None:
                liked_value = 0 
            else: liked_value = 1
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
                sessions[(user_id, article_id)] = self.create_feedback_matrix_session(session, article, session_duration, liked_value, difficulty_feedback_value, article_topic_list)
            else:
                sessions[(user_id, article_id)].session_duration += session_duration

        return self.get_sessions_data(sessions)


    def create_feedback_matrix_session(self, session, article, session_duration, liked_value, difficulty_feedback_value, article_topic_list):
        return FeedbackMatrixSession(
            session.user_id,
            session.article_id,
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
    
    def get_sessions_data(self, sessions: 'dict[tuple[int, int], FeedbackMatrixSession]'):
        '''Manipulate data for each session in the sessions dict, according to the parameters given in the config.'''
        liked_sessions = []
        feedback_diff_list = []
        have_read_sessions = 0

        if self.config.adjustment_config is None:
            self.config.adjustment_config = AdjustmentConfig(difficulty_weight=self.default_difficulty_weight, translation_adjustment_value=self.default_translation_adjustment_value)

        for session in sessions.keys():
            sessions[session].session_duration = self.get_translation_adjustment(sessions[session], self.config.adjustment_config.translation_adjustment_value)
            sessions[session].session_duration = get_difficulty_adjustment(sessions[session], self.config.adjustment_config.difficulty_weight)

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

    def get_translation_adjustment(self, session: FeedbackMatrixSession, adjustment_value):
        timesTranslated = UserActivityData.translated_words_for_article(session.user_id, session.article_id)
        return session.session_duration - (timesTranslated * adjustment_value)

    def duration_is_within_bounds(self, duration, lower, upper):
        return duration <= upper and duration >= lower

    def sessions_to_order_sessions(self, sessions: 'list[FeedbackMatrixSession]'):
        '''Convert user and article ids of sessions to the order defined in our maps'''
        liked_sessions = sessions
        for i in range(len(liked_sessions)):
            liked_sessions[i].user_id = self.user_id_to_order.get(liked_sessions[i].user_id)
            liked_sessions[i].article_id = self.article_id_to_order.get(liked_sessions[i].article_id)
        return liked_sessions

    def generate_dfs(self):
        sessions, liked_sessions, have_read_sessions, feedback_diff_list = self.get_sessions()

        df = self.__session_map_to_df(sessions)
        if self.config.test_tensor:
            liked_df = self.__session_list_to_df([FeedbackMatrixSession(1, 1, 1, 1, 1, 1, [1], 1, 1, 1, 1), FeedbackMatrixSession(1, 5, 1, 1, 1, 1, [1], 1, 1, 1, 1), FeedbackMatrixSession(2, 5, 100, 5, 5, 100, [1], 1, 1, 1, 20)])
        else:
            liked_df = self.__session_list_to_df(liked_sessions)

        self.sessions_df = df
        self.liked_sessions_df = liked_df
        self.have_read_sessions = have_read_sessions
        self.feedback_diff_list_toprint = feedback_diff_list

    def __session_map_to_df(self, sessions: 'dict[tuple[int, int], FeedbackMatrixSession]'):
        data = {index: vars(session) for index, session in sessions.items()}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df

    def __session_list_to_df(self, sessions: 'list[FeedbackMatrixSession]'):
        # Pretty weird logic. We convert a list to a dict and then to a dataframe. Should be changed.
        data = {index: vars(session) for index, session in enumerate(sessions)}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df

    def build_sparse_tensor(self, force=False):
        # This function is not run in the constructor because it takes such a long time to run.
        print("Building sparse tensor")
        if (self.liked_sessions_df is None or self.sessions_df is None or self.have_read_sessions is None) or force:
            self.generate_dfs()

        self.tensor = build_liked_sparse_tensor(self.liked_sessions_df, self.max_user_id, self.max_article_id)

    def plot_sessions_df(self, name):
        print("Plotting sessions. Saving to file: " + name + ".png")
        self.visualizer.plot_urs_with_duration_and_word_count(self.sessions_df, self.have_read_sessions, name, self.config.show_data, self.config.data_since)

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