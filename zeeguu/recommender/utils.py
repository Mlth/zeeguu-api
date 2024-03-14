from datetime import datetime
import os
from enum import Enum, auto
from zeeguu.core.model.user_language import UserLanguage
from zeeguu.core.model.user_reading_session import UserReadingSession
from zeeguu.core.model.user import User
from zeeguu.core.model.article import Article


resource_path = os.path.dirname(os.path.abspath(__file__)) + "/resources/"
average_reading_speed = 70
upper_bound_reading_speed = 45
lower_bound_reading_speed = -35

accurate_duration_date = datetime(day=30, month=1, year=2024)

class ShowData(Enum):
    '''If no ShowData is chosen, all data will be retrieved and shown.'''
    LIKED = auto()
    RATED_DIFFICULTY = auto()

def get_resource_path():
    if not os.path.exists(resource_path):
        os.makedirs(resource_path)
        print(f"Folder '{resource_path}' created successfully.")
    return resource_path

def get_expected_reading_time(word_count, offset):
    ''' The higher the offset is, the higher we want the WPM to be. When WPM is larger, the user is expected to be able to read faster.
     Thus, high offset/WPM = low expected reading time. '''
    return (word_count / (average_reading_speed + offset)) * 60

def cefr_to_fk_difficulty(number):
    result = 0

    if 0 <= number <= 20:
        result = 1
    elif 21 <= number <= 40:
        result = 2
    elif 41 <= number <= 60:
        result = 3
    elif 61 <= number <= 80:
        result = 4
    elif 81 <= number <= 100:
        result = 5

    # This implementation matches the information that Oscar found online. This gives some weird results because a lot of articles are above 50.
    '''if 0 <= number <= 10:
        result = 1
    elif 11 <= number <= 20:
        result = 2
    elif 21 <= number <= 30:
        result = 3
    elif 31 <= number <= 40:
        result = 4
    elif 41 <= number <= 50:
        result = 5
    elif 51 <= number <= 100:
        result = 6'''

    return result

def get_diff_in_article_and_user_level(article_diff, user_level, weight):
    if article_diff > user_level:
        diff = 1 + (((article_diff - user_level) / 100) * weight)
    elif article_diff < user_level:
        diff = 1 - (((user_level - article_diff) / 100) * weight)
    else:
        diff = 1

    return diff

def days_since_normalizer(days_since):
        if days_since < 365 * 1/4:
            return 1
        elif days_since < 365 * 2/4:
            return 0.75
        elif days_since < 365 * 3/4:
            return 0.5
        return 0.25

def add_filters_to_query(query, show_data: list[ShowData]):
    or_filters = []
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
    if len(or_filters) > 0:
        query = query.filter(or_(*or_filters))
    return query


def get_user_reading_sessions(data_since: datetime, show_data: list[ShowData] = []):
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
            .order_by(UserReadingSession.user_id.asc())
    )
    if data_since:
        query = query.filter(UserReadingSession.start_time >= data_since)
    
    return add_filters_to_query(query, show_data).all()


def get_difficulty_adjustment(session, weight):
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