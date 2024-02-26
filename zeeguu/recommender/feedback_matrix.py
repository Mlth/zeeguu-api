from zeeguu.core.model.article import Article
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
tf = tf.compat.v1
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

class FeedbackMatrix:
    tensor = None
    sessions_df = None
    liked_sessions_df = None
    have_read_sessions = None

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

    def get_sessions_with_likes(self):
        print("Getting sessions with likes")
        sessions = {}
        liked_sessions = []
        have_read_sessions = 0

        query_data = self.get_all_user_reading_sessions()

        for session in query_data:
            article_id = session.article_id
            user_id = session.user_id
            article = Article.find_by_id(article_id)
            session_duration = int(session.duration) / 1000 # in seconds
            liked = UserArticle.query.filter_by(user_id=user_id, article_id=article_id).with_entities(UserArticle.liked).first()
            liked_value = 0 if liked == (False,) or liked is None else 1 # should check out
            
            if (user_id, article_id) not in sessions:
                sessions[(user_id, article_id)] = {
                    'user_id': user_id,
                    'article_id': article_id,
                    'user_duration': session_duration,
                    'language': article.language_id,
                    'difficulty': article.fk_difficulty,
                    'word_count': article.word_count,
                    'liked': liked_value,
                    'days_since': (datetime.now() - session.start_time).days,
                }
            else:
                sessions[(user_id, article_id)]['user_duration'] += session_duration
        
        for session in sessions.keys():
            user_id = session[0]
            article_id = session[1]

            should_spend_reading_lower_bound = get_expected_reading_time(sessions[session]['word_count'], 20)
            should_spend_reading_upper_bound = get_expected_reading_time(sessions[session]['word_count'], -20)

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
            
            if userDurationWithTranslated <= should_spend_reading_upper_bound and userDurationWithTranslated >= should_spend_reading_lower_bound and sessions[session]['liked'] == 0:
                have_read_sessions += 1
                sessions[session]['liked'] = 1
            elif sessions[session]['liked'] == 1:
                have_read_sessions += 1
                liked_sessions.append(sessions[session])
            else:
                continue


        return sessions, liked_sessions, have_read_sessions

    def plot_urs_with_duration_and_word_count(self, df, have_read_sessions, file_name):
        x_min, x_max = 0, 2000
        y_min, y_max = 0, 2000

        plt.xlabel('Word count')
        plt.ylabel('Duration')

        dot_color = np.where(df['liked'] == 1, 'blue', 'red')
        plt.scatter(df['word_count'], df['user_duration'], alpha=[self.__days_since_to_multiplier(d) for d in df['days_since']], color=dot_color)

        x_values = df['word_count']
        y_values_line = [get_expected_reading_time(x, -20) for x in x_values]
        plt.plot(x_values, y_values_line, color='red', label='y = ')

        x_values = df['word_count']
        y_values_line = [get_expected_reading_time(x, 20) for x in x_values]
        plt.plot(x_values, y_values_line, color='red', label='y = ')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)
        plt.rc('axes', axisbelow=True)

        have_read_ratio = have_read_sessions / len(df) * 100
        have_not_read_ratio = 100 - have_read_ratio
        plt.text(0, 1.1, f"Have liked: {have_read_ratio:.2f}%", transform=plt.gca().transAxes)
        plt.text(0, 1.05, f"Have not liked: {have_not_read_ratio:.2f}%", transform=plt.gca().transAxes)

        plt.savefig(file_name + '.png')
        print("Saving file: " + file_name + ".png")
        plt.show()

    def plot_sessions_df(self, name):
        print("Plotting sessions. Saving to file: " + name + ".png")
        self.plot_urs_with_duration_and_word_count(self.sessions_df, self.have_read_sessions, name)

    def calc_dfs(self):
        sessions, liked_sessions, have_read_sessions = self.get_sessions_with_likes()
        df = self.__session_map_to_df(sessions)
        liked_df = self.__session_list_to_df(liked_sessions)

        self.sessions_df = df
        self.liked_sessions_df = liked_df
        self.have_read_sessions = have_read_sessions

    def build_sparse_tensor(self, force=False):
        # This function is not run in the constructor because it takes such a long time to run. 
        # Also, the feedback matrix is only updated during training, so we need to be able to explicitly state when it should be called.
        if (self.liked_sessions_df is None or self.sessions_df is None or self.have_read_sessions is None) or force:
            self.calc_dfs()
        indices = self.sessions_df[['user_id', 'article_id']].values
        values = self.sessions_df['liked'].values
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
        df = pd.DataFrame(sessions, columns=['user_id', 'article_id', 'user_duration', 'language', 'difficulty', 'word_count', 'liked', 'days_since'])
        return df

    def print_tensor(self):
        print(tf.sparse.to_dense(self.tensor))

    def __days_since_to_multiplier(self, days_since):
        if days_since < 365 * 1/4:
            return 1
        elif days_since < 365 * 2/4:
            return 0.75
        elif days_since < 365 * 3/4:
            return 0.5
        elif days_since < 365:
            return 0.25
        return 0