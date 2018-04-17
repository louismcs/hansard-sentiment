""" Contains the code for parsing the speeches from the Hansard """

import re
import json
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import Levenshtein
import database

from dates import date_range
from dates import START_DATE
from dates import END_DATE


def remove_title(name):
    """ Removes the first title prefix from the given string """
    titles = ['Mr ', 'Ms ', 'Mrs ', 'Miss ', 'Dr ', 'Professor ',
              'Reverend ', 'Sir ', 'Dame ', 'Hon. ', 'Hon ']

    for title in titles:
        if name.startswith(title):
            name = name[len(title):]

    return name


def remove_titles(name):
    """ Removes title prefixes from the given string """
    ret = remove_title(name)
    while ret != name:
        name = ret
        ret = remove_title(name)

    return ret


def match_first_and_family_name(no_titles, name_list, speaker):
    """ Returns the member id for mps using their given and family names """
    max_similarity = 0
    for name in name_list:
        similarity = Levenshtein.ratio('{} {}'.format(name[2], name[3]), no_titles)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = name

    if max_similarity > 0.85:
        name_list.remove(best_match)
        return best_match[0]
    else:
        with open("blacklist.txt", "a") as myfile:
            myfile.write("{};{};{}\n".format(speaker, best_match[1], best_match[0]))
        raise database.MatchException()


def match_full_name(speaker, member_lists):
    """ Returns the member_id for the given speaker where a match exists """
    if speaker in member_lists['match_list']:
        mp_id = member_lists['match_list'][speaker]
    else:
        max_similarity = 0
        no_titles = remove_titles(speaker)

        for name in member_lists['name_list']:
            similarity = Levenshtein.ratio(remove_titles(name[1]), no_titles)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name

        if max_similarity > 0.85:
            member_lists['name_list'].remove(best_match)
            mp_id = best_match[0]
        else:
            mp_id = match_first_and_family_name(no_titles, member_lists['name_list'], speaker)

        member_lists['match_list'][speaker] = mp_id

    return mp_id

def get_speeches():
    """ Adds debates and their speeches to the database """
    with database.Database() as corpus:
        name_list = corpus.generate_name_list()

        match_list = {'Earl of Ancram' : '259',
                      'Dr Jenny Tonge' : '200',
                      'Mr Nigel Evans' : '474',
                      'Mr Mick Clapham' : '388',
                      'Miss Margaret Jackson' : '328',
                      'Mr Archy Kirkwood' : '635',
                      'Reverend Martin Smyth' : '644',
                      'Mr Andrew Stunell' : '445',
                      'Dr Jack Cunningham' : '496',
                      'Mr Jim Paice' : '124',
                      'Mr Tony Banks' : '3748',
                      'Mr Quentin Davies' : '346',
                      'Ms Helen Brinton' : '122',
                      'Mr Richard Allan' : '397',
                      'Mr Tim Boswell' : '352',
                      'Mr Phil Willis' : '4151',
                      'Lady Sylvia Hermon' : '1437',
                      'Mr Lindsay Hoyle' : '467',
                      'Mr Michael Spicer' : '270',
                      'Mr Tony Wright' : '125',
                      'Mr John Austin-Walker' : '168',
                      'Mr Jim Knight' : '4160',
                      'Mr Bill Cash' : '288',
                      'Mr Chris Smith' : '186',
                      'Mr Eddie O\'Hara' : '482',
                      'Mr Don Touhig' : '542',
                      'Mr Michael Wills' : '2819',
                      'Mr Andy Love' : '164',
                      'Mr David Trimble' : '658',
                      'Mrs Ann Taylor' : '407',
                      'Mr Des Browne' : '620',
                      'Dr Des Turner' : '23',
                      'Mrs Irene Adams' : '631',
                      'Mr Nick Brown' : '523',
                      'Mrs Anne Picking' : '1410',
                      'Ms Dawn Primarolo' : '217',
                     }

        member_lists = {
            'name_list': name_list,
            'match_list': match_list
        }

        for day in date_range(START_DATE, END_DATE):
            add_day(corpus, day, member_lists)


def get_paragraph_text(paragraph):
    """Converts a paragraph tag to plain text"""
    paragraph = re.sub(r"<p.*?>", "", paragraph)
    paragraph = re.sub(r"</p.*?>", "", paragraph)
    paragraph = re.sub(r"<a.*?>.*?</a>", "", paragraph)
    paragraph = re.sub(r"<span.*?>.*?</span>", "", paragraph)
    paragraph = re.sub(r"<q.*?>.*?</q>", "\"", paragraph)
    return paragraph


def add_quote(corpus, blockquote, url, member_lists):
    """Adds a quote (identified by its html element) to the database"""
    try:
        speaker = blockquote.cite.a['title']
        paragraphs = blockquote.find_all("p")
        quote = ""

        for paragraph in paragraphs:
            quote += get_paragraph_text(str(paragraph)) + "\n"

        try:
            member_id = match_full_name(speaker, member_lists)
            corpus.insert_speech(url, member_id, quote)
        except database.MatchException:
            pass

    except TypeError:
        print('Cannot parse quote')


def add_debate(corpus, url, day, title, member_lists):
    """Adds the speeches from a debate (identified by its url) to the database"""
    corpus.insert_debate(url, day, title)
    print('Debate: {} - {}'.format(title, day.strftime("%Y/%b/%d")))
    page = urlopen(url)
    page_soup = BeautifulSoup(page, "html.parser")
    blockquotes = page_soup.find_all("blockquote")

    for blockquote in blockquotes:
        add_quote(corpus, blockquote, url, member_lists)


def add_day(corpus, day, member_lists):
    """Gets the speeches for a given day"""
    date_string = day.strftime("%Y/%b/%d").lower()
    url = 'http://hansard.millbanksystems.com/sittings/{}.js'.format(date_string)
    res = requests.get(url)

    try:
        obj = json.loads(res.text)
        try:
            sections = obj[0]['house_of_commons_sitting']['top_level_sections']
            for section in sections:
                try:
                    sec = section['section']
                    add_debate(corpus, 'http://hansard.millbanksystems.com/commons/{}/{}'
                               .format(date_string, sec['slug']), day, sec['title'],
                               member_lists)
                except KeyError:
                    print('Not a standard section')
        except KeyError:
            print('Not standard sections')

    except ValueError:
        print('No data for {}'.format(date_string))
