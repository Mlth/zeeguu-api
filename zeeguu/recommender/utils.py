def get_expected_reading_time(word_count, offset):
    # The higher the offset is, the higher we want the WPM to be. When WPM is larger, the user is expected to be able to read faster.
    # Thus, high offset/WPM = low expected reading time.
    return (word_count / (70 + offset)) * 60