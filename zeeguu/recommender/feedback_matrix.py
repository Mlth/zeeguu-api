from zeeguu.core.model.article import Article
from zeeguu.core.model.article_difficulty_feedback import ArticleDifficultyFeedback
from zeeguu.core.model.user import User
from zeeguu.core.model.user_activitiy_data import UserActivityData
from zeeguu.core.model.user_article import UserArticle
from zeeguu.core.model.user_language import UserLanguage
from zeeguu.core.model.user_reading_session import UserReadingSession
from zeeguu.recommender.utils import get_diff_in_article_and_user_level, get_expected_reading_time, switch_difficulty
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from zeeguu.core.model import db
import numpy as np
import tensorflow as tf
from collections import Counter
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class FeedbackMatrix:
    upper_bound_reading_speed = 20
    lower_bound_reading_speed = -20

    tensor = None
    sessions_df = None
    liked_sessions_df = None
    have_read_sessions = None
    feedback_diff_list_toprint = None
    feedback_counter = 0

    def get_all_user_reading_sessions(self):
        print("Getting all user reading sessions")
        query_data = (
            UserReadingSession.query
                .filter(UserReadingSession.article_id.isnot(None))
                .filter(UserReadingSession.duration >= 30000) # 30 seconds
                .filter(UserReadingSession.duration <= 3600000) # 1 hour
                .filter(UserReadingSession.start_time >= datetime.now() - timedelta(days=365)) # 1 year
                .order_by(UserReadingSession.user_id.asc())
                .all()
        )
        return query_data

    def get_sessions(self, adjust=True):
        print("Getting sessions with likes")
        sessions = {}

        query_data = self.get_all_user_reading_sessions()

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
                sessions[(user_id, article_id)] = {
                    'user_id': user_id,
                    'article_id': article_id,
                    'user_duration': session_duration,
                    'language': article.language_id,
                    'difficulty': article.fk_difficulty,
                    'word_count': article.word_count,
                    'topics': article_topic_list,
                    'expected_read': 0,
                    'liked': liked_value,
                    'df_feedback': difficulty_feedback_value,
                    'days_since': (datetime.now() - session.start_time).days,
                }
            else:
                sessions[(user_id, article_id)]['user_duration'] += session_duration

        if adjust:
            return self.adjust_sessions(sessions)
        else:
            return self.no_adjust_sessions(sessions)
        
    def no_adjust_sessions(self, sessions):
        liked_sessions = []
        have_read_sessions = 0
        feedback_diff_list = []

        for session in sessions.keys():
            
            should_spend_reading_lower_bound = get_expected_reading_time(sessions[session]['word_count'], self.upper_bound_reading_speed)
            should_spend_reading_upper_bound = get_expected_reading_time(sessions[session]['word_count'], self.lower_bound_reading_speed)
            
            user_duration = sessions[session]['user_duration']

            if user_duration <= should_spend_reading_upper_bound and user_duration >= should_spend_reading_lower_bound:
                have_read_sessions += 1
                sessions[session]['expected_read'] = 1
                liked_sessions.append(sessions[session])
                feedback_diff_list.append(sessions[session]['df_feedback'])
            else:
                continue

        return sessions, liked_sessions, have_read_sessions, feedback_diff_list

    def adjust_sessions(self, sessions):
        liked_sessions = []
        feedback_diff_list = []
        have_read_sessions = 0

        for session in sessions.keys():
            user_id = session[0]
            article_id = session[1]

            should_spend_reading_lower_bound = get_expected_reading_time(sessions[session]['word_count'], self.upper_bound_reading_speed)
            should_spend_reading_upper_bound = get_expected_reading_time(sessions[session]['word_count'], self.lower_bound_reading_speed)

            user_level = UserLanguage.query.filter_by(user_id = user_id, language_id=sessions[(user_id, article_id)]['language']).filter(UserLanguage.cefr_level.isnot(None)).with_entities(UserLanguage.cefr_level).first()
            if user_level is None or user_level[0] == 0 or user_level[0] is None or user_level[0] == [] or user_level == []:
                usr_lvl = 1
            else:
                usr_lvl = user_level[0]

            diff = sessions[session]['difficulty']
            calcDiff = switch_difficulty(diff)
            diff = get_diff_in_article_and_user_level(calcDiff, usr_lvl)
            timesTranslated = UserActivityData.translated_words_for_article(user_id, article_id)
            userDurationWithTranslated = (sessions[session]['user_duration'] - (timesTranslated * 3)) * diff
            sessions[session]['user_duration'] = userDurationWithTranslated 
            
            if userDurationWithTranslated <= should_spend_reading_upper_bound and userDurationWithTranslated >= should_spend_reading_lower_bound:
                have_read_sessions += 1
                sessions[session]['expected_read'] = 1
                liked_sessions.append(sessions[session])
                feedback_diff_list.append(sessions[session]['df_feedback'])
            else:
                continue

        return sessions, liked_sessions, have_read_sessions, feedback_diff_list

    # ------------------- PLOTTING -------------------

    def get_diff_color(self, df, precise=False):
        if precise:
            return np.where(df['df_feedback'] == 1, 'yellow', np.where(df['df_feedback'] == 3, 'blue', 'black'))
        else:
            return "yellow"

    def plot_urs_with_duration_and_word_count(self, df, have_read_sessions, file_name, simple=False):
        x_min, x_max = 0, 2000
        y_min, y_max = 0, 2000

        plt.xlabel('Word count')
        plt.ylabel('Duration')

        expected_read_color = np.where(df['liked'] == 1, 'green', 
                                    np.where(df['df_feedback'] != 0, self.get_diff_color(df, simple),
                                        np.where(df['expected_read'] == 1, 'blue', 'red')))
        plt.scatter(df['word_count'], df['user_duration'], alpha=[self.__days_since_to_multiplier(d) for d in df['days_since']], color=expected_read_color)

        x_values = df['word_count']
        y_values_line = [get_expected_reading_time(x, self.lower_bound_reading_speed) for x in x_values]
        plt.plot(x_values, y_values_line, color='red', label='y = ')

        x_values = df['word_count']
        y_values_line = [get_expected_reading_time(x, self.upper_bound_reading_speed) for x in x_values]
        plt.plot(x_values, y_values_line, color='red', label='y = ')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)
        plt.rc('axes', axisbelow=True)

        if have_read_sessions > 0:
            have_read_ratio = have_read_sessions / len(df) * 100
            have_not_read_ratio = 100 - have_read_ratio
            plt.text(0, 1.1, f"Have read: {have_read_ratio:.2f}%", transform=plt.gca().transAxes)
            plt.text(0, 1.05, f"Have not read: {have_not_read_ratio:.2f}%", transform=plt.gca().transAxes)
            plt.text(0, 1.01, f"Green = liked, yellow = easy, blue = Ok, black = Difficult", transform=plt.gca().transAxes)
        if not simple:
            plt.text(0, 1.01, f"Green = liked, yellow = easy, blue = Ok, black = Difficult", transform=plt.gca().transAxes)

        #Change to '.svg' and format to 'svg' for svg.
        plt.savefig(file_name + '.png', format='png', dpi=900)
        print("Saving file: " + file_name + ".png")
        plt.show()

    

    def plot_sessions_df(self, name):
        print("Plotting sessions. Saving to file: " + name + ".png")
        self.plot_urs_with_duration_and_word_count(self.sessions_df, self.have_read_sessions, name)

    def plot_difficulty_sessions_df(self, name):
        print("Plotting difficulty sessions. Saving to file: " + name + ".png")
        print("Printing the amount of difficulty feedback recored. Saving to file: " + name + ".txt")

        self.print_feedback_difficulty_list(name)
        self.plot_urs_with_duration_and_word_count(self.sessions_df[self.sessions_df['df_feedback'] != 0], self.have_read_sessions, name, True)

    # ------------------- END PLOTTING -------------------

    def generate_dfs(self, adjustment_value=True):
        sessions, liked_sessions, have_read_sessions, feedback_diff_list = self.get_sessions(adjustment_value)
        df = self.__session_map_to_df(sessions)
        liked_df = self.__session_list_to_df(liked_sessions)
        

        self.sessions_df = df
        self.liked_sessions_df = liked_df
        self.have_read_sessions = have_read_sessions
        self.feedback_diff_list_toprint = feedback_diff_list

    def generate_simple_df(self):
        sessions, _, _ = self.get_sessions(False, True, True)
        df = self.__session_map_to_df(sessions)

        self.sessions_df = df

    def build_sparse_tensor(self, force=False):
        # This function is not run in the constructor because it takes such a long time to run.
        print("Building sparse tensor")
        if (self.liked_sessions_df is None or self.sessions_df is None or self.have_read_sessions is None) or force:
            self.generate_dfs()
        indices = self.liked_sessions_df[['user_id', 'article_id']].values
        values = self.liked_sessions_df['expected_read'].values
        num_of_users = User.num_of_users()
        num_of_articles = Article.num_of_articles()
        tensor = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[num_of_users, num_of_articles]
        )
        self.tensor = tensor

    def __session_map_to_df(self, sessions):
        df = pd.DataFrame.from_dict(sessions, orient='index')
        return df

    def __session_list_to_df(self, sessions):
        df = pd.DataFrame(sessions, columns=['user_id', 'article_id', 'user_duration', 'language', 'difficulty', 'word_count', 'expected_read', 'liked', "df_feedback" 'days_since'])
        return df

    def print_tensor(self):
        print(tf.sparse.to_dense(self.tensor))

    def print_feedback_difficulty_list(self, name):

        element_counts = Counter(self.feedback_diff_list_toprint)

        with open(name + ".txt", 'w') as f:
            f.write(f"Total number of feedback recorded from the session: {self.feedback_counter}\n")
            f.write(f"The amount of different values inside the proposed range:\n")
            for element, count in element_counts.items():
                f.write(f"Difficulty: {element}: count: {count}\n")
            f.write(f"The number of feedbacks insde the range: {len(self.feedback_diff_list_toprint)}\n")    
            f.write(f"The number of feedbacks outside the range: {self.feedback_counter - len(self.feedback_diff_list_toprint)}\n")    


    def __days_since_to_multiplier(self, days_since):
        if days_since < 365 * 1/4:
            return 1
        elif days_since < 365 * 2/4:
            return 0.75
        elif days_since < 365 * 3/4:
            return 0.5
        return 0.25