import pickle
from numpy import array, zeros

import database
from trainiraq import get_members_from_file, get_speeches, generate_train_data

def get_settings_and_data():
    settings = {
        'use_test_data': False,
        'black_list': [],
        'white_list': [],
        'bag_size': 500,
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

        settings['testing_mps'] = []

        settings['training_mps'] = (get_members_from_file(settings['train_mp_file']) +
                                    get_members_from_file(settings['test_mp_file']))

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

        settings['speeches'] = get_speeches(corpus, train_data, settings['debates'])

        print('Got speeches')

    return settings, train_data

def find_n_grams():

    #settings, data = get_settings_and_data()

    #features, samples, common_words = generate_train_data(settings, data)

    #features = array([feature['speech_bag'] for feature in features])

    features = pickle.load(open('xfeatures.p', 'rb'))
    samples = pickle.load(open('xsamples.p', 'rb'))
    common_words = pickle.load(open('xwords.p', 'rb'))

    combined_word_counts = zeros(500)
    positive_word_counts = zeros(500)

    for i, feature in enumerate(features):
        combined_word_counts += feature
        if samples[i] == 1:
            positive_word_counts += feature

    positive_probabilities = positive_word_counts / combined_word_counts

    n_gram_data = {}

    for i, n_gram in enumerate(common_words):
        n_gram_data[n_gram] = {
            'probability': positive_probabilities[i],
            'count': combined_word_counts[i]
        }
        print('{},{},{}'.format(combined_word_counts[i], n_gram, positive_probabilities[i]))





find_n_grams()
