""" A collection of functions to retrieve MPs' voting records """

import json
import requests
import database

from dates import date_range
from dates import START_DATE
from dates import END_DATE


def get_division_id(about):
    """ Returns the division id given the about field from the division json """
    return about[36:]


def get_member_id(about):
    """ Returns the member id given the about field from the division json """
    return about[34:]


def get_member_vote(vote_type):
    """ Returns the member vote given the type field from the division json """
    return vote_type[38:]


def get_member_data(member_id, session):
    """ Returns the relevant data for an mp given their id """
    url = 'http://lda.data.parliament.uk/members/{}.json'.format(member_id)
    obj = session.get(url).json()
    primary_topic = obj['result']['primaryTopic']

    try:
        full_name = primary_topic['fullName']['_value']
    except KeyError:
        full_name = ''

    try:
        given_name = primary_topic['givenName']['_value']
    except KeyError:
        given_name = ''

    try:
        additional_name = primary_topic['additionalName']['_value']
    except KeyError:
        additional_name = ''

    try:
        family_name = primary_topic['familyName']['_value']
    except KeyError:
        family_name = ''

    try:
        party = primary_topic['party']['_value']
    except KeyError:
        party = ''

    try:
        constituency = primary_topic['constituency']['label']['_value']
    except KeyError:
        constituency = ''

    ret = {
        'full_name' : full_name,
        'given_name' : given_name,
        'additional_name' : additional_name,
        'family_name' : family_name,
        'party' : party,
        'constituency' : constituency,
    }

    return ret


def get_voting_record():
    """ Creates the database and fills it """

    with database.Database() as corpus:
        corpus.create_tables()

        for day in date_range(START_DATE, END_DATE):
            division_inserts(corpus, day)

        fill_member_and_vote_tables(corpus)


def division_inserts(corpus, day):
    """ Inserts all the divisions for a given day into the database """
    division_date = day.strftime('%Y-%m-%d')
    url = 'http://lda.data.parliament.uk/commonsdivisions.json?date=' \
           + division_date \
           + '&exists-date=true&_view=Commons+Divisions&_pageSize=500&_page=0'
    with requests.Session() as session:
        try:
            obj = session.get(url).json()
        except json.decoder.JSONDecodeError:
            print('JSON ERROR. URL: {}'.format(url))
        divisions = obj['result']['items']

        for division in divisions:
            division_id = get_division_id(division['_about'])
            title = division['title']
            corpus.insert_division(division_id, division_date, title)



def fill_member_and_vote_tables(corpus):
    """ Fills the Member and Vote tables in the database """
    rows = corpus.get_all_division_ids()
    member_ids = []
    with requests.Session() as session:
        for row in rows:
            division_id = row[0]
            print(division_id)
            url = 'http://lda.data.parliament.uk/commonsdivisions/id/{}.json'.format(division_id)
            try:
                obj = session.get(url).json()
                votes = obj['result']['primaryTopic']['vote']

                for vote in votes:
                    member_id = get_member_id(vote['member'][0]['_about'])
                    member_vote = get_member_vote(vote['type'])

                    if member_id not in member_ids:
                        member_ids.append(member_id)
                        member_data = get_member_data(member_id, session)
                        corpus.insert_member(member_id, member_data)
                    corpus.insert_vote(member_id, division_id, member_vote)
            except json.decoder.JSONDecodeError:
                print('JSON ERROR')
