import os
from enum import Enum, auto

resource_path = script_dir = os.path.dirname(os.path.abspath(__file__)) + "/resources/"
average_reading_speed = 70
upper_bound_reading_speed = 45
lower_bound_reading_speed = -35

class ShowData(Enum):
    ALL = auto()
    LIKED = auto()
    RATED_DIFFICULTY = auto()
    NEW_DATA = auto()

def get_expected_reading_time(word_count, offset):
    # The higher the offset is, the higher we want the WPM to be. When WPM is larger, the user is expected to be able to read faster.
    # Thus, high offset/WPM = low expected reading time.
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