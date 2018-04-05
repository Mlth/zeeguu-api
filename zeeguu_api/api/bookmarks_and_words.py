from datetime import datetime
import time

import flask
from flask import request

from zeeguu.model import Bookmark, UserWord, Text, Url, Language
from zeeguu.util.timer_logging_decorator import time_this
from .utils.json_result import json_result
from .utils.route_wrappers import cross_domain, with_session
from . import api, db_session
from zeeguu.default_exercises.default_words import bookmark_data as bd


@api.route("/user_words", methods=["GET"])
@cross_domain
@with_session
def studied_words():
    """
    Returns a list of the words that the user is currently studying.
    """
    return json_result(flask.g.user.user_words())


@api.route("/bookmarks_by_day/<return_context>", methods=["GET"])
@cross_domain
@with_session
def get_bookmarks_by_day(return_context):
    """
    Returns the bookmarks of this user organized by date
    :param return_context: If "with_context" it also returns the
    text where the bookmark was found. If <return_context>
    is anything else, the context is not returned.

    """
    with_context = return_context == "with_context"
    return json_result(flask.g.user.bookmarks_by_day(with_context))


@api.route("/bookmarks_by_day", methods=["POST"])
@cross_domain
@with_session
def post_bookmarks_by_day():
    """
    Returns the bookmarks of this user organized by date. Based on the
    POST arguments, it can return also the context of the bookmark as
    well as it can return only the bookmarks after a given date.

    :param (POST) with_context: If this parameter is "true", the endpoint
    also returns the text where the bookmark was found.

    :param (POST) after_date: the date after which to start retrieving
     the bookmarks. if no date is specified, all the bookmarks are returned.
     The date format is: %Y-%m-%dT%H:%M:%S. E.g. 2001-01-01T01:55:00

    """
    with_context = request.form.get("with_context", "false") == "true"
    after_date_string = request.form.get("after_date", "1970-01-01T00:00:00")
    after_date = datetime.strptime(after_date_string, '%Y-%m-%dT%H:%M:%S')

    return json_result(flask.g.user.bookmarks_by_day(with_context, after_date))


@api.route("/create_default_exercises", methods=["GET"])
@cross_domain
@with_session
@time_this
def create_default_exercises():
    bookmark_datas = [bd['de'], bd['nl'], bd['fr']]

    for bookmark_data in bookmark_datas:
        for example in bookmark_data:
            bookmark = Bookmark.find_or_create(
                db_session,
                flask.g.user,
                example[0], 'de',
                example[1], 'en',
                example[2],
                example[3], 'Url title...')
            db_session.add(bookmark)
            db_session.commit()

    for bookmark in Bookmark.query.all():
        db_session.delete(bookmark)
        db_session.flush()

    for each in UserWord.query.all():
        db_session.delete(each)
        db_session.flush()

    for each in Text.query.all():
        db_session.delete(each)
        db_session.flush()

    for each in Url.query.all():
        db_session.delete(each)
        db_session.flush()

    for each in Language.query.all():
        db_session.delete(each)
        db_session.flush()

    return "OK"