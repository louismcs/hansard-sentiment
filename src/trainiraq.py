""" Functions to parse MP data for use in a classifier """

import collections

from trainhelper import generate_classifier_data, generate_word_list, normalise


def generate_question_bags(settings):
    """ Returns bags of words for the motions specified in the given settings"""
    sum_bag = collections.Counter()
    bags = {}
    for division_id in settings['division_ids']:
        with open('Data/Iraq/motion{}.txt'.format(division_id), 'r') as motion_file:
            motion = motion_file.readlines()[0]
        word_list = generate_word_list(motion, settings)
        bag = collections.Counter()
        for word in word_list:
            sum_bag[word] += 1
            bag[word] += 1
        bags[division_id] = bag

    ret = {}
    for division_id in settings['division_ids']:
        if settings['normalise']:
            ret[division_id] = normalise([bags[division_id][word] for word in sum_bag])
        else:
            ret[division_id] = [bags[division_id][word] for word in sum_bag]

    return ret


def fetch_speeches(speeches, mp_data):
    """ Given a list of information about mps, returns all of
        the speeches made by these mps in the given speeches"""

    ret = []

    for member in mp_data:
        ret = ret + speeches[member['id']]

    return ret


def parse_speeches(settings, mp_data, train):
    """ Parses MPs' speeches and creates the corresponding bags of words """
    if settings['entailment']:
        question_bags = generate_question_bags(settings)

    speeches = fetch_speeches(settings['speeches'], mp_data)

    aye_features = []
    no_features = []

    sum_bag = collections.Counter()

    members = {}

    for speech in speeches:
        if speech['member'] not in members:
            members[speech['member']] = speech['votes']
        word_list = generate_word_list(speech['text'], settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
            sum_bag[word] += 1
        if settings['entailment']:
            if train:
                for division_id in settings['division_ids']:
                    if speech['votes'][division_id]:
                        aye_features.append({
                            'speech_bag': bag,
                            'question_bag': question_bags[division_id],
                            'member': speech['member']
                        })
                    else:
                        no_features.append({
                            'speech_bag': bag,
                            'question_bag': question_bags[division_id],
                            'member': speech['member']
                        })
            else:
                if speech['votes'][settings['test_division']]:
                    aye_features.append({
                        'speech_bag': bag,
                        'question_bag': question_bags[settings['test_division']],
                        'member': speech['member']
                    })
                else:
                    no_features.append({
                        'speech_bag': bag,
                        'question_bag': question_bags[settings['test_division']],
                        'member': speech['member']
                    })
        else:
            if speech['votes'][settings['test_division']]:
                aye_features.append({
                    'speech_bag': bag,
                    'member': speech['member']
                })
            else:
                no_features.append({
                    'speech_bag': bag,
                    'member': speech['member']
                })


    return aye_features, no_features, sum_bag, members


def generate_train_data(settings, mp_list):
    """ Returns the features and samples in a form that can be used
        by a classifier, given the filenames for the data """

    aye_features, no_features, sum_bag, _ = parse_speeches(settings, mp_list, True)
    if settings['max_bag_size']:
        common_words = [word[0] for word in sum_bag.most_common(settings['bag_size'])]
    else:
        common_words = []
        for term in sum_bag:
            if sum_bag[term] > 0:
                common_words.append(term)

    features, samples = generate_classifier_data(aye_features, no_features, common_words,
                                                 settings['normalise'])

    return features, samples, common_words


def generate_test_data(common_words, settings, mp_list):
    """ Returns the features and samples in a form that can be used
        by a classifier, given the filenames for the data """

    aye_features, no_features, _, members = parse_speeches(settings, mp_list, False)

    features, samples = generate_classifier_data(aye_features, no_features, common_words,
                                                 settings['normalise'])

    return features, samples, members


def count_ayes(speeches):
    """ Given a list of speeches and the predictions of their
        stances, returns the number of aye predictions """

    ret = 0
    for speech in speeches:
        if speech['prediction'] == 1:
            ret += 1

    return ret


def get_members_from_file(file_path):
    """ Returns a list of ids of MPs to be reserved for testing given the file path """

    with open(file_path) as id_file:
        ret = [line.rstrip() for line in id_file]

    return ret


def get_speeches(corpus, member_list, debates):
    """ Returns all the speeches in the given database that match the given settings """
    speeches = {}
    for member in member_list:
        speeches[member['id']] = []
        for debate in debates:
            speeches[member['id']] = speeches[member['id']] + corpus.get_speech_texts(member,
                                                                                      debate)

    return speeches
