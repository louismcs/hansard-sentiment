import database

from random import shuffle
from numpy import mean
from numpy import sqrt
from numpy import std



def get_member_ids(corpus, debate_terms, division_id):
    """ Given a list of terms, finds all the debates whose titles contain one or more of these terms and returns their ids """

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(corpus.get_members_from_term(term, division_id)))

    return debates


def intersect_member_ids(corpus, debate_terms, division_ids):
    """ Given a list of terms, finds all the debates whose titles contain one or more of these terms and returns their ids """
    member_sets = [corpus.get_member_ids(debate_terms, division_id) for division_id in division_ids]
    return list(set.intersection(*member_sets))


def get_debate_ids(corpus, debate_terms):
    """ Returns a list of debate ids matching the given settings """

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(corpus.get_debates_from_term(term)))
    return list(debates)


def choose_test_data(corpus, debate_terms, division_ids):
    member_ids = intersect_member_ids(corpus, debate_terms, division_ids)
    test_size = round(0.1 * len(member_ids))
    debate_ids = get_debate_ids(corpus, debate_terms)
    aye_percents_in_range = False
    no_of_speeches_in_range = False
    speech_counts = {}
    aye_votes = {}
    for member_id in member_ids:
        speech_counts[member_id] = corpus.get_member_no_of_speeches(debate_ids, member_id)
        aye_votes[member_id] = {}
        for division_id in division_ids:
            aye_votes[member_id][division_id] = corpus.is_aye_vote(division_id, member_id)
    while not (aye_percents_in_range and no_of_speeches_in_range):
        shuffle(member_ids)
        test_ids = member_ids[:test_size]
        train_ids = [member_id for member_id in member_ids if member_id not in test_ids]

        test_ayes = {}
        test_percents = {}
        train_ayes = {}
        train_percents = {}
        total_ayes = {}
        total_percents = {}

        aye_percents_in_range = True
        for division_id in division_ids:
            total_ayes[division_id] = 0
            test_ayes[division_id] = 0
            for test_id in test_ids:
                if aye_votes[test_id][division_id]:
                    test_ayes[division_id] += 1
                    total_ayes[division_id] += 1

            test_percents[division_id] = 100 * test_ayes[division_id] / len(test_ids)
            train_ayes[division_id] = 0
            for train_id in train_ids:
                if aye_votes[train_id][division_id]:
                    train_ayes[division_id] += 1
                    total_ayes[division_id] += 1

            train_percents[division_id] = 100 * train_ayes[division_id] / len(train_ids)

            total_percents[division_id] = 100 * total_ayes[division_id] / len(member_ids)

            aye_percents_in_range = (aye_percents_in_range
                                     and test_percents[division_id] > total_percents[division_id] - 5
                                     and test_percents[division_id] < total_percents[division_id] + 5)

        num_of_speeches = []
        total_test_speeches = 0
        for member_id in member_ids:
            speech_number = speech_counts[member_id]
            num_of_speeches.append(speech_number)
            if member_id in test_ids:
                total_test_speeches += speech_number

        mean_test_speeches = total_test_speeches / len(test_ids)

        mean_total_speeches = mean(num_of_speeches)

        std_total_speeches = std(num_of_speeches) / sqrt(len(member_ids))

        no_of_speeches_in_range = (mean_test_speeches > mean_total_speeches - std_total_speeches
                                   and mean_test_speeches < mean_total_speeches + std_total_speeches)

    print('Overall mean no. of speeches: {}'.format(mean_total_speeches))
    print('Test mean no. of speeches:    {}'.format(mean_test_speeches))

    for division_id in division_ids:
        print()
        print('{} overall aye percentage: {}%'.format(division_id, total_percents[division_id]))
        print('{} test aye percentage:     {}%'.format(division_id, test_percents[division_id]))

    test_file = open('test-stratified.txt', 'w')

    for test_id in test_ids:
        test_file.write('{}\n'.format(test_id))

    train_file = open('train-stratified.txt', 'w')

    for train_id in train_ids:
        train_file.write('{}\n'.format(train_id))


def split():
    corpus = database.Database()
    terms = ['iraq', 'terrorism', 'middle east',
             'defence policy', 'defence in the world', 'afghanistan']
    division_ids = [102564, 102565]
    choose_test_data(corpus, terms, division_ids)
