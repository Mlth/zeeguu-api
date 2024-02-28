average_reading_speed = 70

def get_expected_reading_time(word_count, offset):
    # The higher the offset is, the higher we want the WPM to be. When WPM is larger, the user is expected to be able to read faster.
    # Thus, high offset/WPM = low expected reading time.
    return (word_count / (average_reading_speed + offset)) * 60

def switch_difficulty(number):
    result = 0

    if 0 <= number <= 10:
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
        result = 6

    return result

def get_diff_in_article_and_user_level(article_diff, user_level):
    if article_diff > user_level:
        diff = 1 + ((article_diff - user_level) / 10)
    elif article_diff < user_level:
        diff = 1 - ((user_level - article_diff) / 10)
    else:
        diff = 1

    return diff