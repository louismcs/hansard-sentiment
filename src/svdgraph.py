import pickle

from numpy import array
from sklearn.manifold import TSNE

from matplotlib import pyplot

import database
from trainiraq import get_members_from_file, get_speeches, generate_train_data, generate_test_data


def get_settings_and_data():
    settings = {
        'use_test_data': False,
        'black_list': [],
        'white_list': [],
        'bag_size': 100,
        'max_bag_size': True,
        'remove_stopwords': False,
        'stem_words': False,
        'group_numbers': False,
        'n_gram': 2,
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

    return settings, data


def generate_features(settings, data):
    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    return train_features, test_features, train_samples, test_samples


def make_svd_scatter():

    #settings, data = get_settings_and_data()

    """ settings = pickle.load(open('svdsettings.p', 'rb'))

    data = pickle.load(open('svddata.p', 'rb'))

    settings['max_bag_size'] = True
    settings['bag_size'] = 500

    train_features, test_features, train_samples, test_samples = generate_features(settings, data)

    train_features = array([feature['speech_bag'] for feature in train_features])
    test_features = array([feature['speech_bag'] for feature in test_features])

    TSNE(n_components=2).fit_transform(train_features)

    print(train_features.shape)

    print('reduced features')

    print(train_features[0])

    pickle.dump(train_features, open('features.p', 'wb'))
    pickle.dump(train_samples, open('samples.p', 'wb')) """

    train_features = pickle.load(open('features.p', 'rb'))
    train_samples = pickle.load(open('samples.p', 'rb'))
    

    positive_xs = []
    positive_ys = []
    negative_xs = []
    negative_ys = []

    print(train_samples[0])
    for i, sample in enumerate(train_samples):
        if sample == 1:
            positive_xs.append(train_features[i][0])
            positive_ys.append(train_features[i][1])
        else:
            negative_xs.append(train_features[i][0])
            negative_ys.append(train_features[i][1])

    print('Got coords')

    """ ax = pyplot.gca()
    ax.set_xscale('log')
    ax.set_yscale('log') """
    pyplot.scatter(positive_xs, positive_ys, s=1, marker="P")
    pyplot.scatter(negative_xs, negative_ys, s=1, marker="X")

    pyplot.show()

make_svd_scatter()
