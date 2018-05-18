""" Functions to train and test an SVM classifier on MPs' speeches on the Iraq war """

import pickle
from random import shuffle
from numpy import array, array_split, mean, sqrt, std
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

import database
from trainhelper import (generate_linear_param_sets,
                         generate_linear_values, generate_poly_param_sets,
                         generate_poly_values, generate_rbf_param_sets,
                         generate_rbf_values, generate_refined_linear_values,
                         generate_refined_poly_values,
                         generate_refined_rbf_values,
                         reduce_features, generate_lower_log_params, generate_higher_log_params,
                         generate_lower_params, generate_higher_params)
from trainiraq import (generate_train_data, generate_test_data, count_ayes, get_members_from_file,
                       get_speeches)





def get_all_member_ids(corpus, debate_terms, division_id):
    """ Returns a list of all the member ids corresponding to members who voted
        in the given division and spoke in at least one debate which matches at
        least one of the given debate terms, according to the given database"""

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(corpus.get_all_members_from_term(term, division_id)))

    return list(debates)


def get_aye_member_ids(corpus, debate_terms, division_id):
    """ Returns a list of all the member ids corresponding to members who voted 'aye'
        in the given division and spoke in at least one debate which matches at
        least one of the given debate terms, according to the given database"""

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(corpus.get_aye_members_from_term(term, division_id)))

    return list(debates)


def get_no_member_ids(corpus, debate_terms, division_id):
    """ Returns a list of all the member ids corresponding to members who voted 'no'
        in the given division and spoke in at least one debate which matches at
        least one of the given debate terms, according to the given database"""

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(corpus.get_no_members_from_term(term, division_id)))

    return list(debates)


def get_mp_folds(corpus, settings):
    """ Given the number of folds, returns that number of
        non-overlapping lists (of equal/nearly equal length) of
        ids of mps matching the given settings """
    speech_counts = {}
    votes = {}
    speech_count_list = []
    total_ayes = {}
    for division_id in settings['division_ids']:
        total_ayes[division_id] = 0

    for member_id in settings['training_mps']:
        no_of_speeches = corpus.get_member_no_of_speeches(settings['debates'], member_id)
        speech_counts[member_id] = no_of_speeches
        speech_count_list.append(no_of_speeches)
        votes[member_id] = {}
        for division_id in settings['division_ids']:
            vote = corpus.is_aye_vote(division_id, member_id)
            votes[member_id][division_id] = vote
            if vote:
                total_ayes[division_id] += 1

    total_percents = {}
    for division_id in settings['division_ids']:
        total_percents[division_id] = 100 * total_ayes[division_id] / len(settings['training_mps'])

    mean_total_speeches = mean(speech_count_list)
    std_total_speeches = 4 * std(speech_count_list) / sqrt(len(speech_count_list))
    aye_percents_in_range = False
    no_of_speeches_in_range = False
    member_data = []
    for member in settings['training_mps']:
        member_data.append({
            'id': member,
            'votes': votes[member]
        })
    while not (aye_percents_in_range and no_of_speeches_in_range):
        shuffle(member_data)

        test_folds = [list(element) for element in array_split(member_data,
                                                               settings['no_of_folds'])]

        aye_percents_in_range = True
        no_of_speeches_in_range = True
        for fold in test_folds:
            fold_speech_count = 0

            for member in fold:
                fold_speech_count += speech_counts[member['id']]

            mean_fold_speech_count = fold_speech_count / len(fold)
            no_of_speeches_in_range = (no_of_speeches_in_range
                                       and mean_fold_speech_count >
                                       mean_total_speeches - std_total_speeches
                                       and mean_fold_speech_count <
                                       mean_total_speeches + std_total_speeches)

            for division_id in settings['division_ids']:
                aye_votes = 0
                for member in fold:
                    if member['votes'][division_id]:
                        aye_votes += 1

                aye_percent = 100 * aye_votes / len(fold)
                aye_percents_in_range = (aye_percents_in_range
                                         and aye_percent > total_percents[division_id] - 15
                                         and aye_percent < total_percents[division_id] + 15)

    ret = []
    for test_fold in test_folds:
        train = [member for member in member_data if member not in test_fold]
        ret.append({
            'test': test_fold,
            'train': train
        })

    return ret


def get_complete_bags(features, entailment):
    """ Given the features and a boolean which determines whether or not
        entailment is being applied, returns the bags of words to be used  """

    if entailment:
        ret = [feature['speech_bag'] + feature['question_bag'] for feature in features]
    else:
        ret = [feature['speech_bag'] for feature in features]

    return ret


def compute_linear_fold_f1s(settings, data, linear_param_values):
    """ Given the settings, the testing set, the training set and a list of sets of
        hyperparameters, trains an SVM classifier with a linear kernel for each of
        the sets of hyperparameters on the training data and returns the F1 scores
        for each of the sets """

    print('Doing linear fold')

    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    complete_train_features = array(get_complete_bags(train_features, settings['entailment']))

    complete_test_features = array(get_complete_bags(test_features, settings['entailment']))

    if settings['svd']:
        complete_train_features, complete_test_features = reduce_features(complete_train_features,
                                                                          complete_test_features)

    ret = []

    for param_values in linear_param_values:
        classifier = svm.SVC(C=param_values['c'], kernel='linear', cache_size=settings['cache'])

        print('Before')
        classifier.fit(complete_train_features, train_samples)
        print('After')
        test_predictions = classifier.predict(complete_test_features)

        score = f1_score(test_samples, test_predictions)

        ret.append(score)

        print('{} / {} F1: {}. C = {}'.format(len(ret), len(linear_param_values), score,
                                              param_values['c']))

    return ret


def compute_rbf_fold_f1s(settings, data, rbf_param_values):
    """ Given the settings, the testing set, the training set and a list of sets of
        hyperparameters, trains an SVM classifier with an rbf kernel for each of
        the sets of hyperparameters on the training data and returns the F1 scores
        for each of the sets """

    print('Doing rbf fold')
    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    complete_train_features = array(get_complete_bags(train_features, settings['entailment']))

    complete_test_features = array(get_complete_bags(test_features, settings['entailment']))

    if settings['svd']:
        complete_train_features, complete_test_features = reduce_features(complete_train_features,
                                                                          complete_test_features)

    ret = []

    for param_values in rbf_param_values:
        classifier = svm.SVC(C=param_values['c'], kernel='rbf', gamma=param_values['gamma'],
                             cache_size=settings['cache'])

        classifier.fit(complete_train_features, train_samples)

        test_predictions = classifier.predict(complete_test_features)

        score = f1_score(test_samples, test_predictions)
        ret.append(score)

        print('{} / {} F1: {}'.format(len(ret), len(rbf_param_values), score))

    return ret


def compute_poly_fold_f1s(settings, data, poly_param_values):
    """ Given the settings, the testing set, the training set and a list of sets of
        hyperparameters, trains an SVM classifier with a poly kernel for each of
        the sets of hyperparameters on the training data and returns the F1 scores
        for each of the sets """

    print('Doing poly fold')
    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    complete_train_features = array(get_complete_bags(train_features, settings['entailment']))

    complete_test_features = array(get_complete_bags(test_features, settings['entailment']))

    if settings['svd']:
        complete_train_features, complete_test_features = reduce_features(complete_train_features,
                                                                          complete_test_features)

    ret = []

    for param_values in poly_param_values:
        classifier = svm.SVC(C=param_values['c'], kernel='poly', degree=param_values['d'],
                             gamma=param_values['d'], coef0=param_values['r'],
                             cache_size=settings['cache'])

        classifier.fit(complete_train_features, train_samples)

        test_predictions = classifier.predict(complete_test_features)
        score = f1_score(test_samples, test_predictions)
        ret.append(score)

        print('{} / {} F1: {}'.format(len(ret), len(poly_param_values), score))

    return ret


def compute_f1(settings, data):
    """ Computes the F1 score for given data according to the given settings """

    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    if settings['kernel'] == 'default':
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

    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''



    complete_train_features = array(get_complete_bags(train_features, settings['entailment']))
    complete_test_features = array(get_complete_bags(test_features, settings['entailment']))

    if settings['svd']:
        complete_train_features, complete_test_features = reduce_features(complete_train_features,
                                                                          complete_test_features)

    classifier.fit(complete_train_features, train_samples)
    test_predictions = classifier.predict(complete_test_features)

    return f1_score(test_samples, test_predictions)


def compute_member_f1s(settings, data):
    """ Runs one loop of the cross-validation """

    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, members = generate_test_data(common_words, settings, data['test'])

    if settings['kernel'] == 'linear':
        classifier = svm.SVC(C=settings['linear_c'], kernel='linear', cache_size=settings['cache'])
    elif settings['kernel'] == 'rbf':
        classifier = svm.SVC(C=settings['rbf_c'], kernel='rbf', gamma=settings['rbf_gamma'],
                             cache_size=settings['cache'])
    else:
        #Assert kernel is poly
        classifier = svm.SVC(C=settings['poly_c'], kernel='poly', degree=settings['poly_d'],
                             gamma=settings['poly_gamma'], coef0=settings['poly_r'],
                             cache_size=settings['cache'])

    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''

    complete_train_features = get_complete_bags(train_features, settings['entailment'])
    complete_test_features = get_complete_bags(test_features, settings['entailment'])

    if settings['svd']:
        complete_train_features, complete_test_features = reduce_features(complete_train_features,
                                                                          complete_test_features)

    classifier.fit(complete_train_features, train_samples)

    test_predictions = classifier.predict(complete_test_features)
    for pred in test_predictions:
        if pred == -1:
            print('aaa')
    grouped_speeches = {}

    for member_id in settings['testing_mps']:
        grouped_speeches[member_id] = {
            'votes': members[member_id],
            'speeches': []
        }

    for i, feature in enumerate(test_features):
        grouped_speeches[feature['member']]['speeches'].append({
            'feature': complete_test_features[i],
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


def find_linear_params(settings, mp_folds, linear_param_values):
    """ Given some settings, a set of cross-validation folds and a list of sets
        of linear kernel hyperparameters, performs an exhaustive search over
        the hyperparameters to determine the optimal hyperparameters """

    params_found = False
    while not params_found:
        print('Linear loop')
        params_found = True
        linear_param_sets = generate_linear_param_sets(linear_param_values)

        f1_matrix = array([compute_linear_fold_f1s(settings, fold, linear_param_sets)
                           for fold in mp_folds]).transpose()

        max_f1_mean = -1

        for i, param_f1s in enumerate(f1_matrix):
            f1_mean = mean(param_f1s)
            if f1_mean > max_f1_mean:
                multiple_max_means = False
                max_linear_param_set = linear_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)
            elif f1_mean == max_f1_mean:
                multiple_max_means = True

        if len(linear_param_values['cs']) > 1 and not multiple_max_means:
            min_c = linear_param_values['cs'][0]
            max_c = linear_param_values['cs'][len(linear_param_values['cs']) - 1]

            if max_linear_param_set['c'] == min_c:
                c_log_diff = linear_param_values['cs'][1] / linear_param_values['cs'][0]
                linear_param_values['cs'] = generate_lower_log_params(min_c, 5, c_log_diff)
                params_found = False
            elif max_linear_param_set['c'] == max_c:
                c_log_diff = linear_param_values['cs'][1] / linear_param_values['cs'][0]
                linear_param_values['cs'] = generate_higher_log_params(max_c, 5, c_log_diff)
                params_found = False

    return max_f1s, max_linear_param_set


def find_rbf_params(settings, mp_folds, rbf_param_values):
    """ Given some settings, a set of cross-validation folds and a list of sets
        of rbf kernel hyperparameters, performs an exhaustive search over
        the hyperparameters to determine the optimal hyperparameters """

    params_found = False
    while not params_found:
        print('Rbf loop')
        params_found = True
        rbf_param_sets = generate_rbf_param_sets(rbf_param_values)

        f1_matrix = array([compute_rbf_fold_f1s(settings, fold, rbf_param_sets)
                           for fold in mp_folds]).transpose()

        max_f1_mean = -1

        for i, param_f1s in enumerate(f1_matrix):
            f1_mean = mean(param_f1s)
            if f1_mean > max_f1_mean:
                multiple_max_means = False
                max_rbf_param_set = rbf_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)
            elif f1_mean == max_f1_mean:
                multiple_max_means = True

        if not multiple_max_means:
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
                    params_found = False

    return max_f1s, max_rbf_param_set


def find_poly_params(settings, mp_folds, poly_param_values):
    """ Given some settings, a set of cross-validation folds and a list of sets
        of poly kernel hyperparameters, performs an exhaustive search over
        the hyperparameters to determine the optimal hyperparameters """

    params_found = False
    while not params_found:
        print('Poly loop')
        params_found = True
        poly_param_sets = generate_poly_param_sets(poly_param_values)

        f1_matrix = array([compute_poly_fold_f1s(settings, fold, poly_param_sets)
                           for fold in mp_folds]).transpose()

        max_f1_mean = -1

        for i, param_f1s in enumerate(f1_matrix):
            f1_mean = mean(param_f1s)
            if f1_mean > max_f1_mean:
                multiple_max_means = False
                max_poly_param_set = poly_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)
            elif f1_mean == max_f1_mean:
                multiple_max_means = True

        if not multiple_max_means:
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
                    params_found = False

    return max_f1s, max_poly_param_set


def change_n_gram(settings, increment, current_f1s, mp_folds):
    """ Determines the optimal n-gram value for the classifier """

    significant_change = settings['n_gram'] + increment in range(1, 10)

    while significant_change:
        settings['n_gram'] += increment
        new_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]

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


def choose_boolean_setting(settings, setting, current_f1s, mp_folds):
    """ Determines the optimal value of a given boolean setting for the classifier """

    settings[setting] = True
    new_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]

    current_mean = mean(current_f1s)
    new_mean = mean(new_f1s)

    if new_mean > current_mean:
        current_f1s = new_f1s
    else:
        settings[setting] = False

    return current_f1s


def refine_linear_params(settings, mp_folds):
    """ Finds the optimal set of linear kernel hyperparameters in a finely grained search """
    linear_param_values = generate_linear_values(5)

    _, linear_params = find_linear_params(settings, mp_folds, linear_param_values)

    linear_param_values = generate_refined_linear_values(linear_params, 7)

    return find_linear_params(settings, mp_folds, linear_param_values)


def refine_rbf_params(settings, mp_folds):
    """ Finds the optimal set of rbf kernel hyperparameters in a finely grained search """

    rbf_param_values = generate_rbf_values(5, 7)

    _, rbf_params = find_rbf_params(settings, mp_folds, rbf_param_values)

    rbf_param_values = generate_refined_rbf_values(rbf_params, 7, 7)

    return find_rbf_params(settings, mp_folds, rbf_param_values)


def refine_poly_params(settings, mp_folds):
    """ Finds the optimal set of poly kernel hyperparameters in a finely grained search """
    poly_param_values = generate_poly_values(4, 1, 1, 1)

    _, poly_params = find_poly_params(settings, mp_folds, poly_param_values)

    poly_param_values = generate_refined_poly_values(poly_params, 7, 7, 1, 1)

    return find_poly_params(settings, mp_folds, poly_param_values)


def learn_settings(settings, mp_folds):
    """ Determines the optimal set of settings and hyperparameters for the classifier """

    current_f1s = [compute_f1(settings, fold) for fold in mp_folds]

    current_f1s = choose_boolean_setting(settings, 'normalise', current_f1s, mp_folds)

    current_f1s = choose_boolean_setting(settings, 'remove_stopwords', current_f1s, mp_folds)

    current_f1s = choose_boolean_setting(settings, 'stem_words', current_f1s, mp_folds)

    current_f1s = choose_boolean_setting(settings, 'group_numbers', current_f1s, mp_folds)

    print('Boolean settings learned')
    print(current_f1s)

    current_f1s = change_n_gram(settings, 1, current_f1s, mp_folds)

    print('N gram learned')
    print(current_f1s)
    linear_param_values = generate_linear_values(5)

    linear_f1s, linear_params = find_linear_params(settings, mp_folds, linear_param_values)

    linear_mean = mean(linear_f1s)

    print('Linear mean: {}'.format(linear_mean))

    rbf_param_values = generate_rbf_values(5, 7)

    rbf_f1s, rbf_params = find_rbf_params(settings, mp_folds, rbf_param_values)

    rbf_mean = mean(rbf_f1s)

    print('RBF mean: {}'.format(rbf_mean))

    poly_param_values = generate_poly_values(4, 1, 1, 1)

    poly_f1s, poly_params = find_poly_params(settings, mp_folds, poly_param_values)

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

    if settings['kernel'] == 'linear':
        current_f1s, linear_params = refine_linear_params(settings, mp_folds)
        settings['linear_c'] = linear_params['c']

    elif settings['kernel'] == 'rbf':
        current_f1s, rbf_params = refine_rbf_params(settings, mp_folds)
        settings['rbf_c'] = rbf_params['c']
        settings['rbf_gamma'] = rbf_params['gamma']

    else:
        #Assert kernel is poly
        current_f1s, poly_params = refine_poly_params(settings, mp_folds)
        settings['poly_c'] = poly_params['c']
        settings['poly_gamma'] = poly_params['gamma']
        settings['poly_d'] = poly_params['d']
        settings['poly_r'] = poly_params['r']

    pickle.dump(settings, open('settings.p', 'wb'))

    print('Hyper parameters refined')
    print(current_f1s)
    print('Average CV F1: {} ± {}'.format(mean(current_f1s), std(current_f1s)))


def run():
    """ Sets the settings and runs the program """

    settings = {
        'use_test_data': False,
        'black_list': [],
        'white_list': [],
        'bag_size': 500,
        'max_bag_size': False,
        'remove_stopwords': False,
        'stem_words': False,
        'group_numbers': False,
        'n_gram': 1,
        'test_division': 102565,
        'all_debates': False,
        'debate_terms': ['iraq', 'terrorism', 'middle east', 'defence policy',
                         'defence in the world', 'afghanistan'],
        'no_of_folds': 10,
        'entailment': True,
        'normalise': False,
        'svd': True,
        'division_ids': [102564, 102565],
        'test_mp_file': 'test-stratified.txt',
        'train_mp_file': 'train-stratified.txt',
        'cache': 1024,
        'kernel': 'default'
    }


    with database.Database() as corpus:

        settings['debates'] = corpus.get_debates(settings)

        settings['testing_mps'] = get_members_from_file(settings['test_mp_file'])

        settings['training_mps'] = get_members_from_file(settings['train_mp_file'])

        mp_folds = get_mp_folds(corpus, settings)

        print('Made splits')

        train_data = mp_folds[0]['test'] + mp_folds[0]['train']

        test_data = []

        for member in settings['testing_mps']:
            votes = {}
            for division_id in settings['division_ids']:
                votes[division_id] = corpus.is_aye_vote(division_id, member)
            test_data.append({
                'id': member,
                'votes': votes
            })

        member_data = train_data + test_data

        settings['speeches'] = get_speeches(corpus, member_data, settings['debates'])

    print('Got speeches')

    learn_settings(settings, mp_folds)

    print('Normalisation: {}'.format(settings['normalise']))
    print('N-gram: {}'.format(settings['n_gram']))
    print('Remove stopwords: {}'.format(settings['remove_stopwords']))
    print('Stem words: {}'.format(settings['stem_words']))
    print('Group numbers: {}'.format(settings['group_numbers']))
    print()

    if settings['use_test_data']:
        data = {
            'train': train_data,
            'test': test_data
        }
        settings['max_bag_size'] = False
        compute_member_f1s(settings, data)


def run_from_settings_file():
    """ Sets the settings from a file and runs the program """

    settings = pickle.load(open('settings.p', 'rb'))
    settings['normalise'] = True
    settings['use_test_data'] = True
    with database.Database() as corpus:
        mp_folds = get_mp_folds(corpus, settings)

        print('Made splits')

        train_data = mp_folds[0]['test'] + mp_folds[0]['train']

        test_data = []

        for member in settings['testing_mps']:
            votes = {}
            for division_id in settings['division_ids']:
                votes[division_id] = corpus.is_aye_vote(division_id, member)
            test_data.append({
                'id': member,
                'votes': votes
            })

    data = {
        'train': train_data,
        'test': test_data
    }

    compute_member_f1s(settings, data)


run_from_settings_file()
