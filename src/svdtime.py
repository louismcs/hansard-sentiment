import time
import pickle
from sklearn import svm
from sklearn.metrics import f1_score
import database
from trainiraq import get_members_from_file, get_speeches, generate_train_data, generate_test_data
from trainhelper import reduce_features

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

    train_features = [feature['speech_bag'] for feature in train_features]
    test_features = [feature['speech_bag'] for feature in test_features]

    return train_features, test_features, train_samples, test_samples


def time_reduction():

    #settings, data = get_settings_and_data()

    settings = pickle.load(open('svdsettings.p', 'rb'))

    data = pickle.load(open('svddata.p', 'rb'))

    start = time.time()
    train_features, test_features, train_samples, test_samples = generate_features(settings, data)
    end = time.time()
    print('Simple reduction time: {} seconds'.format(end - start))

    classifier = svm.SVC()
    classifier.fit(train_features, train_samples)

    test_predictions = classifier.predict(test_features)

    print(f1_score(test_samples, test_predictions))

    settings['bag_size'] = 2000

    train_features, test_features, train_samples, test_samples = generate_features(settings, data)
    start = time.time()
    reduce_features(train_features, test_features)
    end = time.time()
    print('SVD reduction time: {} seconds'.format(end - start))

    classifier = svm.SVC()
    classifier.fit(train_features, train_samples)

    test_predictions = classifier.predict(test_features)

    print(f1_score(test_samples, test_predictions))

time_reduction()
