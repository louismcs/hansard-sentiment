""" Functions to train and test a naive Bayes classifier on MPs' speeches on the Iraq war """

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm

import database

from trainiraq import generate_train_data
from trainiraq import generate_test_data
from trainiraq import count_ayes
from trainiraq import get_members_from_file
from trainiraq import get_speeches



def compute_member_f1s(settings, data):
    """ Runs one loop of the cross-validation """

    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, members = generate_test_data(common_words, settings, data['test'])

    #classifier = svm.SVC()
    
    #classifier = MLPClassifier(solver='lbfgs')

    classifier = GaussianNB()

    classifier.fit([feature['speech_bag'] for feature in train_features], train_samples)

    test_predictions = classifier.predict([feature['speech_bag'] for feature in test_features])

    grouped_speeches = {}

    for member_id in settings['testing_mps']:
        grouped_speeches[member_id] = {
            'votes': members[member_id],
            'speeches': []
        }

    for i, feature in enumerate(test_features):
        grouped_speeches[feature['member']]['speeches'].append({
            'feature': test_features[i],
            'prediction': test_predictions[i]
        })

    member_votes = []

    member_predictions = []

    for member_id in settings['testing_mps']:
        grouped_speeches[member_id]['aye_fraction'] = (
            count_ayes(grouped_speeches[member_id]['speeches']) /
            len(grouped_speeches[member_id]['speeches']))
        grouped_speeches[member_id]['overall_prediction'] = (
            1 if grouped_speeches[member_id]['aye_fraction'] > 0.5 else -1)
        member_votes.append(1 if grouped_speeches[member_id]['votes'][settings['test_division']]
                            else -1)
        member_predictions.append(grouped_speeches[member_id]['overall_prediction'])

    print('Accuracy by MP: {}%'.format(100 * accuracy_score(member_votes, member_predictions)))
    print('F1 by MP: {}'.format(f1_score(member_votes, member_predictions)))

    print('Accuracy by speech: {}%'.format(100 * accuracy_score(test_samples, test_predictions)))
    print('F1 by speech: {}'.format(f1_score(test_samples, test_predictions)))

    return f1_score(test_samples, test_predictions)


def run():
    """ Trains and tests a naive Bayes classifier on MPs' data """

    settings = {
        'use_test_data': False,
        'black_list': [],
        'white_list': [],
        'bag_size': 500,
        'max_bag_size': True,
        'remove_stopwords': False,
        'stem_words': False,
        'group_numbers': False,
        'n_gram': 1,
        'test_division': 102565,
        'all_debates': False,
        'debate_terms': ['iraq', 'terrorism', 'middle east', 'defence policy',
                         'defence in the world', 'afghanistan'],
        'no_of_folds': 10,
        'entailment': False,
        'normalise': False,
        'svd': False,
        'division_ids': [102565],
        'test_mp_file': 'test-stratified.txt',
        'train_mp_file': 'train-stratified.txt',
        'cache': 1024
    }
    with database.Database() as corpus:

        settings['debates'] = corpus.get_debates(settings)

        settings['testing_mps'] = get_members_from_file(settings['test_mp_file'])

        settings['training_mps'] = get_members_from_file(settings['train_mp_file'])

        print('Got members')

        train_data = []

        for member in settings['training_mps']:
            votes = {}
            for division_id in settings['division_ids']:
                votes[division_id] = corpus.is_aye_vote(division_id, member)
            train_data.append({
                'id': member,
                'votes': votes
            })

        print('Got training data')

        test_data = []

        for member in settings['testing_mps']:
            votes = {}
            for division_id in settings['division_ids']:
                votes[division_id] = corpus.is_aye_vote(division_id, member)
            test_data.append({
                'id': member,
                'votes': votes
            })

        print('Got test data')


        member_data = train_data + test_data

        settings['speeches'] = get_speeches(corpus, member_data, settings['debates'])

        print('Got speeches')

    data = {
        'train': train_data,
        'test': test_data
    }
    compute_member_f1s(settings, data)

run()
