""" Contains the code to split the data into a training set and a testing set """

from random import shuffle
from numpy import mean
from numpy import sqrt
from numpy import std

import database


def get_member_ids(corpus, debate_terms, division_id):
    """ Given a list of terms, finds all the debates whose titles
        contain one or more of these terms and returns their ids """

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(corpus.get_members_from_term(term, division_id)))

    return debates


def intersect_member_ids(corpus, debate_terms, division_ids):
    """ Given a list of terms, finds all the debates whose titles
        contain one or more of these terms and returns their ids """
    member_sets = [corpus.get_member_ids(debate_terms, division_id) for division_id in division_ids]
    return list(set.intersection(*member_sets))


def get_debate_ids(corpus, debate_terms):
    """ Returns a list of debate ids matching the given settings """

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(corpus.get_debates_from_term(term)))
    return list(debates)


def get_member_stats(corpus, member_ids, debate_ids, division_ids):
    """ Gets the number of speeches and aye votes for each member """

    speech_counts = {}
    aye_votes = {}

    for member_id in member_ids:
        speech_counts[member_id] = corpus.get_member_no_of_speeches(debate_ids, member_id)
        aye_votes[member_id] = {}

        for division_id in division_ids:
            aye_votes[member_id][division_id] = corpus.is_aye_vote(division_id, member_id)

    return {
        'speech_counts': speech_counts,
        'aye_votes': aye_votes
    }


def get_data_in_range(member_ids, test_size, division_ids, member_stats):
    """ Repeatedly generates a testing set until it is appropriately stratified """
    loop_params = {
        'aye_percents_in_range': False,
        'no_of_speeches_in_range': False
    }
    while not (loop_params['aye_percents_in_range'] and loop_params['no_of_speeches_in_range']):
        shuffle(member_ids)
        ids = {
            'test': member_ids[:test_size],
            'train': [member_id for member_id in member_ids
                      if member_id not in member_ids[:test_size]]
        }

        stats = {
            'test_ayes': {},
            'test_percents': {},
            'train_ayes': {},
            'train_percents': {},
            'total_ayes': {},
            'total_percents': {}
        }

        loop_params['aye_percents_in_range'] = True

        for division_id in division_ids:
            stats['total_ayes'][division_id] = 0
            stats['test_ayes'][division_id] = 0

            for test_id in ids['test']:
                if member_stats['aye_votes'][test_id][division_id]:
                    stats['test_ayes'][division_id] += 1
                    stats['total_ayes'][division_id] += 1

            stats['test_percents'][division_id] = (100 * stats['test_ayes'][division_id]
                                                   / len(ids['test']))
            stats['train_ayes'][division_id] = 0

            for train_id in ids['train']:
                if member_stats['aye_votes'][train_id][division_id]:
                    stats['train_ayes'][division_id] += 1
                    stats['total_ayes'][division_id] += 1

            stats['train_percents'][division_id] = (100 * stats['train_ayes'][division_id]
                                                    / len(ids['train']))

            stats['total_percents'][division_id] = (100 * stats['total_ayes'][division_id]
                                                    / len(member_ids))

            loop_params['aye_percents_in_range'] = (loop_params['aye_percents_in_range']
                                                    and stats['test_percents'][division_id] >
                                                    stats['total_percents'][division_id] - 5
                                                    and stats['test_percents'][division_id] <
                                                    stats['total_percents'][division_id] + 5)

        stats['num_of_speeches'] = []
        stats['total_test_speeches'] = 0

        for member_id in member_ids:
            speech_number = member_stats['speech_counts'][member_id]
            stats['num_of_speeches'].append(speech_number)

            if member_id in ids['test']:
                stats['total_test_speeches'] += speech_number

        stats['mean_test_speeches'] = stats['total_test_speeches'] / len(ids['test'])

        stats['mean_total_speeches'] = mean(stats['num_of_speeches'])

        stats['std_total_speeches'] = std(stats['num_of_speeches']) / sqrt(len(member_ids))

        loop_params['no_of_speeches_in_range'] = (stats['mean_test_speeches'] >
                                                  (stats['mean_total_speeches']
                                                   - stats['std_total_speeches'])
                                                  and stats['mean_test_speeches'] <
                                                  (stats['mean_total_speeches']
                                                   + stats['std_total_speeches']))

    return {
        'means': {
            'total_speeches': stats['mean_total_speeches'],
            'test_speeches': stats['mean_test_speeches']
        },
        'percents': {
            'total': stats['total_percents'],
            'test': stats['test_percents']
        }
    }, ids



def choose_test_data(corpus, debate_terms, division_ids):
    """ Determines the testing set  """

    member_ids = intersect_member_ids(corpus, debate_terms, division_ids)
    test_size = round(0.1 * len(member_ids))
    debate_ids = get_debate_ids(corpus, debate_terms)
    member_stats = get_member_stats(corpus, member_ids, debate_ids, division_ids)

    stats, ids = get_data_in_range(member_ids, test_size, division_ids,
                                   member_stats)

    print('Overall mean no. of speeches: {}'.format(stats['means']['total_speeches']))
    print('Test mean no. of speeches:    {}'.format(stats['means']['test_speeches']))

    for division_id in division_ids:
        print('\n{} overall aye percentage: {}%'.format(division_id,
                                                        stats['percents']['total'][division_id]))
        print('{} test aye percentage:     {}%'.format(division_id,
                                                       stats['percents']['test'][division_id]))

    test_file = open('test-stratified.txt', 'w')

    for test_id in ids['test']:
        test_file.write('{}\n'.format(test_id))

    train_file = open('train-stratified.txt', 'w')

    for train_id in ids['train']:
        train_file.write('{}\n'.format(train_id))


def split():
    """ Sets the debate and division variables and calls the
        function to generate the training/testing data split """

    terms = ['iraq', 'terrorism', 'middle east',
             'defence policy', 'defence in the world', 'afghanistan']
    division_ids = [102564, 102565]
    with database.Database() as corpus:
        choose_test_data(corpus, terms, division_ids)
