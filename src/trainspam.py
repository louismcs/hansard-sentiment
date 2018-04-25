""" Learns to detect spam using the spam email dataset """

import collections
import pickle

from random import shuffle

from bs4 import BeautifulSoup
from numpy import array, mean, std
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

from trainhelper import (generate_linear_param_sets, generate_linear_values,
                         generate_poly_param_sets, generate_poly_values,
                         generate_rbf_param_sets, generate_rbf_values,
                         generate_refined_linear_values, generate_refined_rbf_values,
                         generate_refined_poly_values,
                         get_n_grams, normalise, reduce_features,
                         remove_punctuation, remove_stopwords, remove_tags,
                         stem_words)


def generate_word_list(body, settings):
    """ Returns a list of words given an email body """
    body = remove_tags(body)
    body = remove_punctuation(body)
    body = body.lower()

    word_list = body.split()

    if settings['remove_stopwords']:
        word_list = remove_stopwords(word_list, settings['black_list'], settings['white_list'])

    if settings['stem_words']:
        word_list = stem_words(word_list)

    return get_n_grams(word_list, settings['n_gram'])


def get_messages(file):
    """ Returns the body of all messages in the given .ems file """
    handler = open(file).read()
    soup = BeautifulSoup(handler, "lxml-xml")
    return [str(message.MESSAGE_BODY) for message in soup.find_all("MESSAGE")]


def generate_bags(messages, settings, sum_bag):
    """ Given settings and a list of messages, generates a bag of
        words for each, as well as a combined bag of words """

    bags = []
    for message in messages:
        word_list = generate_word_list(message, settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
            sum_bag[word] += 1
        bags.append(bag)

    return bags, sum_bag



def condense_bags(bags, words):
    """ Returns an array of integer arrays containing the counts of the words
         (in the array provided) and an array of the Counter bags """
    return [[bag[word] for word in words] for bag in bags]


def normalise_features(features):
    """ Given a list of features, returns the l2 norms of the same features """

    return [normalise(feature) for feature in features]


def generate_classifier_data(gen_bags, spam_bags, common_words, normalise_data):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the bags and most common words in them """
    features = condense_bags(gen_bags + spam_bags, common_words)
    if normalise_data:
        features = normalise_features(features)

    samples = []
    for _ in range(len(gen_bags)):
        samples.append(1)

    for _ in range(len(spam_bags)):
        samples.append(-1)

    return features, samples


def generate_train_data(train_data, settings):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the data and settings """

    gen_bags, sum_bag = generate_bags(train_data['gen'], settings, collections.Counter())
    spam_bags, sum_bag = generate_bags(train_data['spam'], settings, sum_bag)

    if settings['max_bag_size']:
        common_words = [word[0] for word in sum_bag.most_common(settings['bag_size'])]
    else:
        common_words = []
        for term in sum_bag:
            if sum_bag[term] > 3:
                common_words.append(term)

    features, samples = generate_classifier_data(gen_bags, spam_bags, common_words,
                                                 settings['normalise'])

    return features, samples, common_words


def generate_test_data(test_data, settings, common_words):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """
    gen_bags, _ = generate_bags(test_data['gen'], settings, collections.Counter())
    spam_bags, _ = generate_bags(test_data['spam'], settings, collections.Counter())

    features, samples = generate_classifier_data(gen_bags, spam_bags, common_words,
                                                 settings['normalise'])

    return features, samples


def split_array(messages, no_of_folds):
    """ Splits the given messages list into a given number of roughly equal disjoint lists """

    split_size = round(len(messages) / no_of_folds)
    folds = []
    index = 0
    for _ in range(no_of_folds - 1):
        folds.append(messages[index : index + split_size])
        index += split_size
    folds.append(messages[index:])
    return folds


def make_folds(messages, no_of_folds):
    """ Given a list of messages, splits them into a given number of disjoint lists of messages """

    shuffle(messages)

    test_folds = split_array(messages, no_of_folds)

    ret = []
    for test_fold in test_folds:
        train_fold = [message for message in messages if message not in test_fold]
        ret.append({
            'test': test_fold,
            'train': train_fold
        })

    return ret



def compute_f1(settings, data):
    """ Trains a classifier using the giving training data according to given
        settings and tests the classifier on the given testing data, returning the f1 score """

    train_features, train_samples, common_words = generate_train_data(data['train'], settings)

    test_features, test_samples = generate_test_data(data['test'], settings, common_words)
    if settings['default_svm']:
        classifier = svm.SVC(cache_size=settings['cache'])
    elif settings['kernel'] == 'linear':
        classifier = svm.SVC(C=settings['linear_c'], kernel='linear', cache_size=settings['cache'])
    elif settings['kernel'] == 'rbf':
        classifier = svm.SVC(C=settings['rbf_c'], kernel='rbf', gamma=settings['rbf_gamma'],
                             cache_size=settings['cache'])
    else:
        #Assert kernel is poly
        classifier = svm.SVC(C=settings['poly_c'], kernel='poly', degree=settings['poly_d'],
                             gamma=settings['poly_gamma'], coef0=settings['poly_r'],
                             cache_size=settings['cache'])

    if settings['svd']:
        train_features, test_features = reduce_features(train_features, test_features)

    classifier.fit(train_features, train_samples)

    test_predictions = classifier.predict(test_features)

    print('Accuracy: {}%'.format(100 * accuracy_score(test_samples, test_predictions)))
    print('F1: {}'.format(f1_score(test_samples, test_predictions)))

    return f1_score(test_samples, test_predictions)

def compute_linear_fold_f1s(settings, spam_message_fold, gen_message_fold, linear_param_sets):
    """ Given the settings, the testing set, the training set and a list of sets of
        hyperparameters, trains an SVM classifier with a linear kernel for each of
        the sets of hyperparameters on the training data and returns the F1 scores
        for each of the sets """

    print('Doing linear fold')

    data = {
        'train': {
            'spam': spam_message_fold['train'],
            'gen': gen_message_fold['train']
        },
        'test': {
            'spam': spam_message_fold['test'],
            'gen': gen_message_fold['train']
        }
    }

    train_features, train_samples, common_words = generate_train_data(data['train'], settings)

    test_features, test_samples = generate_test_data(data['train'], settings, common_words)

    if settings['svd']:
        train_features, test_features = reduce_features(train_features, test_features)

    ret = []

    for param_values in linear_param_sets:
        classifier = svm.SVC(C=param_values['c'], kernel='linear', cache_size=settings['cache'])

        classifier.fit(train_features, train_samples)

        test_predictions = classifier.predict(test_features)

        score = f1_score(test_samples, test_predictions)

        ret.append(score)

        print('{} / {} F1: {}. C = {}'.format(len(ret), len(linear_param_sets), score,
                                              param_values['c']))

    return ret


def compute_rbf_fold_f1s(settings, spam_message_fold, gen_message_fold, rbf_param_sets):
    """ Given the settings, the testing set, the training set and a list of sets of
        hyperparameters, trains an SVM classifier with a linear kernel for each of
        the sets of hyperparameters on the training data and returns the F1 scores
        for each of the sets """

    print('Doing rbf fold')

    data = {
        'train': {
            'spam': spam_message_fold['train'],
            'gen': gen_message_fold['train']
        },
        'test': {
            'spam': spam_message_fold['test'],
            'gen': gen_message_fold['train']
        }
    }

    train_features, train_samples, common_words = generate_train_data(data['train'], settings)

    test_features, test_samples = generate_test_data(data['train'], settings, common_words)

    if settings['svd']:
        train_features, test_features = reduce_features(train_features, test_features)

    ret = []

    for param_values in rbf_param_sets:
        classifier = svm.SVC(C=param_values['c'], kernel='rbf', gamma=param_values['gamma'],
                             cache_size=settings['cache'])

        classifier.fit(train_features, train_samples)

        test_predictions = classifier.predict(test_features)

        score = f1_score(test_samples, test_predictions)

        ret.append(score)

        print('{} / {} F1: {}'.format(len(ret), len(rbf_param_sets), score))

    return ret


def compute_poly_fold_f1s(settings, spam_message_fold, gen_message_fold, poly_param_sets):
    """ Given the settings, the testing set, the training set and a list of sets of
        hyperparameters, trains an SVM classifier with a linear kernel for each of
        the sets of hyperparameters on the training data and returns the F1 scores
        for each of the sets """

    print('Doing poly fold')

    data = {
        'train': {
            'spam': spam_message_fold['train'],
            'gen': gen_message_fold['train']
        },
        'test': {
            'spam': spam_message_fold['test'],
            'gen': gen_message_fold['train']
        }
    }

    train_features, train_samples, common_words = generate_train_data(data['train'], settings)

    test_features, test_samples = generate_test_data(data['train'], settings, common_words)

    if settings['svd']:
        train_features, test_features = reduce_features(train_features, test_features)

    ret = []

    for param_values in poly_param_sets:
        classifier = svm.SVC(C=param_values['c'], kernel='poly', degree=param_values['d'],
                             gamma=param_values['d'], coef0=param_values['r'],
                             cache_size=settings['cache'])

        classifier.fit(train_features, train_samples)

        test_predictions = classifier.predict(test_features)

        score = f1_score(test_samples, test_predictions)

        ret.append(score)

        print('{} / {} F1: {}'.format(len(ret), len(poly_param_sets), score))

    return ret


def find_linear_params(settings, spam_message_folds, gen_message_folds, linear_param_values):
    """ Given some settings, a set of cross-validation folds and a list of sets
        of linear kernel hyperparameters, performs an exhaustive search over
        the hyperparameters to determine the optimal hyperparameters """

    params_found = False
    while not params_found:
        print('Linear loop')
        params_found = True
        linear_param_sets = generate_linear_param_sets(linear_param_values)

        f1_matrix = array([compute_linear_fold_f1s(settings, spam_message_folds[i],
                                                   gen_message_folds[i],
                                                   linear_param_sets)
                           for i in range(settings['no_of_folds'])]).transpose()

        max_f1_mean = -1

        for i, param_f1s in enumerate(f1_matrix):
            f1_mean = mean(param_f1s)
            if f1_mean > max_f1_mean:
                #multiple_max_means = False
                max_linear_param_set = linear_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)
            """ elif f1_mean == max_f1_mean:
                multiple_max_means = True """

        """ if len(linear_param_values['cs']) > 1 and not multiple_max_means:
            min_c = linear_param_values['cs'][0]
            max_c = linear_param_values['cs'][len(linear_param_values['cs']) - 1]

            if max_linear_param_set['c'] == min_c:
                c_log_diff = linear_param_values['cs'][1] / linear_param_values['cs'][0]
                linear_param_values['cs'] = generate_lower_log_params(min_c, 5, c_log_diff)
                params_found = False
            elif max_linear_param_set['c'] == max_c:
                c_log_diff = linear_param_values['cs'][1] / linear_param_values['cs'][0]
                linear_param_values['cs'] = generate_higher_log_params(max_c, 5, c_log_diff)
                params_found = False """

    return max_f1s, max_linear_param_set


def find_rbf_params(settings, spam_message_folds, gen_message_folds, rbf_param_values):
    """ Given some settings, a set of cross-validation folds and a list of sets
        of rbf kernel hyperparameters, performs an exhaustive search over
        the hyperparameters to determine the optimal hyperparameters """

    params_found = False
    while not params_found:
        print('Rbf loop')
        params_found = True
        rbf_param_sets = generate_rbf_param_sets(rbf_param_values)

        f1_matrix = array([compute_rbf_fold_f1s(settings, spam_message_folds[i],
                                                gen_message_folds[i],
                                                rbf_param_sets)
                           for i in range(settings['no_of_folds'])]).transpose()
        max_f1_mean = -1

        for i, param_f1s in enumerate(f1_matrix):
            f1_mean = mean(param_f1s)
            if f1_mean > max_f1_mean:
                #multiple_max_means = False
                max_rbf_param_set = rbf_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)
            """ elif f1_mean == max_f1_mean:
                multiple_max_means = True """

        """ if not multiple_max_means:
            if len(rbf_param_values['cs']) > 1:
                min_c = rbf_param_values['cs'][0]
                max_c = rbf_param_values['cs'][len(rbf_param_values['cs']) - 1]

                if max_rbf_param_set['c'] == min_c:
                    c_log_diff = rbf_param_values['cs'][1] / rbf_param_values['cs'][0]
                    rbf_param_values['cs'] = generate_lower_log_params(min_c, 5, c_log_diff)
                    params_found = False
                elif max_rbf_param_set['c'] == max_c:
                    c_log_diff = rbf_param_values['cs'][1] / rbf_param_values['cs'][0]
                    rbf_param_values['cs'] = generate_higher_log_params(max_c, 5, c_log_diff)
                    params_found = False

            if len(rbf_param_values['gammas']) > 1:
                min_gamma = rbf_param_values['gammas'][0]
                max_gamma = rbf_param_values['gammas'][len(rbf_param_values['gammas']) - 1]

                if max_rbf_param_set['gamma'] == min_gamma:
                    gamma_log_diff = rbf_param_values['gammas'][1] / rbf_param_values['gammas'][0]
                    rbf_param_values['gammas'] = generate_lower_log_params(min_gamma, 5,
                                                                           gamma_log_diff)
                    params_found = False
                elif max_rbf_param_set['gamma'] == max_gamma:
                    gamma_log_diff = rbf_param_values['gammas'][1] / rbf_param_values['gammas'][0]
                    rbf_param_values['gammas'] = generate_higher_log_params(max_gamma, 5,
                                                                            gamma_log_diff)
                    params_found = False """

    return max_f1s, max_rbf_param_set


def find_poly_params(settings, spam_message_folds, gen_message_folds, poly_param_values):
    """ Given some settings, a set of cross-validation folds and a list of sets
        of poly kernel hyperparameters, performs an exhaustive search over
        the hyperparameters to determine the optimal hyperparameters """

    params_found = False
    while not params_found:
        print('Poly loop')
        params_found = True
        poly_param_sets = generate_poly_param_sets(poly_param_values)

        f1_matrix = array([compute_poly_fold_f1s(settings, spam_message_folds[i],
                                                 gen_message_folds[i],
                                                 poly_param_sets)
                           for i in range(settings['no_of_folds'])]).transpose()

        max_f1_mean = -1

        for i, param_f1s in enumerate(f1_matrix):
            f1_mean = mean(param_f1s)
            if f1_mean > max_f1_mean:
                #multiple_max_means = False
                max_poly_param_set = poly_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)
            """ elif f1_mean == max_f1_mean:
                multiple_max_means = True """

        """ if not multiple_max_means:
            if len(poly_param_values['cs']) > 1:
                min_c = poly_param_values['cs'][0]
                max_c = poly_param_values['cs'][len(poly_param_values['cs']) - 1]

                if max_poly_param_set['c'] == min_c:
                    c_log_diff = poly_param_values['cs'][1] / poly_param_values['cs'][0]
                    poly_param_values['cs'] = generate_lower_log_params(min_c, 5, c_log_diff)
                    params_found = False
                elif max_poly_param_set['c'] == max_c:
                    c_log_diff = poly_param_values['cs'][1] / poly_param_values['cs'][0]
                    poly_param_values['cs'] = generate_higher_log_params(max_c, 5, c_log_diff)
                    params_found = False

            if len(poly_param_values['gammas']) > 1:
                min_gamma = poly_param_values['gammas'][0]
                max_gamma = poly_param_values['gammas'][len(poly_param_values['gammas']) - 1]

                if max_poly_param_set['gamma'] == min_gamma:
                    gamma_log_diff = poly_param_values['gammas'][1] / poly_param_values['gammas'][0]
                    poly_param_values['gammas'] = generate_lower_log_params(min_gamma, 5,
                                                                            gamma_log_diff)
                    params_found = False
                elif max_poly_param_set['gamma'] == max_gamma:
                    gamma_log_diff = poly_param_values['gammas'][1] / poly_param_values['gammas'][0]
                    poly_param_values['gammas'] = generate_higher_log_params(max_gamma, 5,
                                                                             gamma_log_diff)
                    params_found = False

            if len(poly_param_values['ds']) > 1:
                min_d = poly_param_values['ds'][0]
                max_d = poly_param_values['ds'][len(poly_param_values['ds']) - 1]

                if max_poly_param_set['d'] == min_d:
                    d_diff = poly_param_values['ds'][1] - poly_param_values['ds'][0]
                    poly_param_values['ds'] = generate_lower_params(min_d, 3, d_diff)
                    params_found = False
                elif max_poly_param_set['d'] == max_d:
                    d_diff = poly_param_values['ds'][1] - poly_param_values['ds'][0]
                    poly_param_values['ds'] = generate_higher_params(max_d, 3, d_diff)
                    params_found = False

            if len(poly_param_values['rs']) > 1:
                max_r = poly_param_values['rs'][len(poly_param_values['rs']) - 1]

                if max_poly_param_set['r'] == max_r:
                    r_diff = poly_param_values['rs'][1] - poly_param_values['rs'][0]
                    poly_param_values['rs'] = generate_higher_params(max_r, 3, r_diff)
                    params_found = False """

    return max_f1s, max_poly_param_set


def choose_boolean_setting(settings, setting, current_f1s, folds):
    """ Determines the optimal value of a given boolean setting for the classifier """

    settings[setting] = True
    new_f1s = [compute_f1(settings, fold) for fold in folds]

    current_mean = mean(current_f1s)
    new_mean = mean(new_f1s)

    if new_mean > current_mean:
        current_f1s = new_f1s
    else:
        settings[setting] = False

    return current_f1s


def change_n_gram(settings, increment, current_f1s, folds):
    """ Determines the optimal n-gram value for the classifier """

    significant_change = settings['n_gram'] + increment in range(1, 10)

    while significant_change:
        settings['n_gram'] += increment
        new_f1s = [compute_f1(settings, fold) for fold in folds]

        current_mean = mean(current_f1s)
        new_mean = mean(new_f1s)

        print('New mean for n = {} is {}'.format(settings['n_gram'], new_mean))
        if new_mean > current_mean:
            current_f1s = new_f1s
            if settings['n_gram'] == 10 or settings['n_gram'] == 1:
                significant_change = False
        else:
            significant_change = False
            settings['n_gram'] -= increment

    return current_f1s


def refine_linear_params(settings, spam_message_folds, gen_message_folds):
    """ Finds the optimal set of linear kernel hyperparameters in a finely grained search """
    linear_param_values = generate_linear_values(5)

    _, linear_params = find_linear_params(settings, spam_message_folds, gen_message_folds,
                                          linear_param_values)

    linear_param_values = generate_refined_linear_values(linear_params, 7)

    return find_linear_params(settings, spam_message_folds, gen_message_folds, linear_param_values)


def refine_rbf_params(settings, spam_message_folds, gen_message_folds):
    """ Finds the optimal set of rbf kernel hyperparameters in a finely grained search """

    rbf_param_values = generate_rbf_values(5, 7)

    _, rbf_params = find_rbf_params(settings, spam_message_folds, gen_message_folds,
                                    rbf_param_values)

    rbf_param_values = generate_refined_rbf_values(rbf_params, 7, 7)

    return find_rbf_params(settings, spam_message_folds, gen_message_folds, rbf_param_values)


def refine_poly_params(settings, spam_message_folds, gen_message_folds):
    """ Finds the optimal set of poly kernel hyperparameters in a finely grained search """
    poly_param_values = generate_poly_values(4, 1, 1, 1)

    _, poly_params = find_poly_params(settings, spam_message_folds, gen_message_folds,
                                      poly_param_values)

    poly_param_values = generate_refined_poly_values(poly_params, 7, 7, 1, 1)

    return find_poly_params(settings, spam_message_folds, gen_message_folds, poly_param_values)


def learn_settings(settings, spam_message_folds, gen_message_folds):
    """ Determines the optimal set of settings and hyperparameters for the classifier """

    folds = []

    for i, spam_fold in enumerate(spam_message_folds):
        folds.append({
            'train': {
                'gen': gen_message_folds[i]['train'],
                'spam': spam_fold['train']
            },
            'test': {
                'gen': gen_message_folds[i]['test'],
                'spam': spam_fold['test']
            }
        })

    current_f1s = [compute_f1(settings, fold) for fold in folds]

    current_f1s = choose_boolean_setting(settings, 'normalise', current_f1s, folds)

    current_f1s = choose_boolean_setting(settings, 'remove_stopwords', current_f1s, folds)

    current_f1s = choose_boolean_setting(settings, 'stem_words', current_f1s, folds)

    print('Boolean settings learned')
    print(current_f1s)

    current_f1s = change_n_gram(settings, 1, current_f1s, folds)

    print('N gram learned')
    print(current_f1s)

    settings['default_svm'] = False

    linear_param_values = generate_linear_values(5)

    linear_f1s, linear_params = find_linear_params(settings, spam_message_folds, gen_message_folds,
                                                   linear_param_values)

    linear_mean = mean(linear_f1s)

    print('Linear mean: {}'.format(linear_mean))

    rbf_param_values = generate_rbf_values(5, 7)

    rbf_f1s, rbf_params = find_rbf_params(settings, spam_message_folds, gen_message_folds,
                                          rbf_param_values)

    rbf_mean = mean(rbf_f1s)

    print('RBF mean: {}'.format(rbf_mean))

    poly_param_values = generate_poly_values(4, 1, 1, 1)

    poly_f1s, poly_params = find_poly_params(settings, spam_message_folds, gen_message_folds,
                                             poly_param_values)

    poly_mean = mean(poly_f1s)

    print('Poly mean: {}'.format(poly_mean))

    if linear_mean > rbf_mean:
        if linear_mean > poly_mean:
            settings['kernel'] = 'linear'
            settings['linear_c'] = linear_params['c']
            current_f1s = linear_f1s
        else:
            settings['kernel'] = 'poly'
            settings['poly_c'] = poly_params['c']
            settings['poly_gamma'] = poly_params['gamma']
            settings['poly_d'] = poly_params['d']
            settings['poly_r'] = poly_params['r']
            current_f1s = poly_f1s
    else:
        if rbf_mean > poly_mean:
            settings['kernel'] = 'rbf'
            settings['rbf_c'] = rbf_params['c']
            settings['rbf_gamma'] = rbf_params['gamma']
            current_f1s = rbf_f1s
        else:
            settings['kernel'] = 'poly'
            settings['poly_c'] = poly_params['c']
            settings['poly_gamma'] = poly_params['gamma']
            settings['poly_d'] = poly_params['d']
            settings['poly_r'] = poly_params['r']
            current_f1s = poly_f1s

    print('Hyper parameters learned')
    print(current_f1s)

    pickle.dump(settings, open('unrefined_settings.p', 'wb'))

    if settings['kernel'] == 'linear':
        current_f1s, linear_params = refine_linear_params(settings, spam_message_folds,
                                                          gen_message_folds)
        settings['linear_c'] = linear_params['c']

    elif settings['kernel'] == 'rbf':
        current_f1s, rbf_params = refine_rbf_params(settings, spam_message_folds, gen_message_folds)
        settings['rbf_c'] = rbf_params['c']
        settings['rbf_gamma'] = rbf_params['gamma']

    else:
        #Assert kernel = poly
        current_f1s, poly_params = refine_poly_params(settings, spam_message_folds,
                                                      gen_message_folds)
        settings['poly_c'] = poly_params['c']
        settings['poly_gamma'] = poly_params['gamma']
        settings['poly_d'] = poly_params['d']
        settings['poly_r'] = poly_params['r']

    settings['bag_size'] = 2000

    pickle.dump(settings, open('refined_settings.p', 'wb'))

    print('Hyper parameters refined')
    print(current_f1s)
    print('Average CV F1: {} ± {}'.format(mean(current_f1s), std(current_f1s)))


def run():
    """ Sets the settings and runs the program  """

    settings = {
        'use_test_data': True,
        'black_list': [],
        'white_list': [],
        'bag_size': 100,
        'max_bag_size': True,
        'remove_stopwords': False,
        'stem_words': False,
        'n_gram': 1,
        'no_of_folds': 10,
        'normalise': False,
        'svd': False,
        'cache': 1024,
        'default_svm': True
    }

    spam_train_messages = get_messages('Data/Spam/train_SPAM.ems')
    gen_train_messages = get_messages('Data/Spam/train_GEN.ems')

    spam_message_folds = make_folds(spam_train_messages, settings['no_of_folds'])
    gen_message_folds = make_folds(gen_train_messages, settings['no_of_folds'])

    learn_settings(settings, spam_message_folds, gen_message_folds)

    if settings['use_test_data']:
        spam_test_messages = get_messages('Data/Spam/test_SPAM.ems')
        gen_test_messages = get_messages('Data/Spam/test_GEN.ems')
        data = {
            'train': {
                'spam': spam_train_messages,
                'gen': gen_train_messages
            },
            'test':  {
                'spam': spam_test_messages,
                'gen': gen_test_messages
            }
        }
        settings['max_bag_size'] = False
        compute_f1(settings, data)

run()
