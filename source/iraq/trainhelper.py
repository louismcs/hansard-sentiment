import math
import re
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import ngrams
from numpy import linalg
from numpy import logspace
from numpy import log10

def remove_punctuation(body):
    """ Removes punctuation from a given string """
    body = body.replace("\n", " ")
    body = re.sub(r"[^\w\d\s#'-]", '', body)
    body = body.replace(" '", " ")
    body = body.replace("' ", " ")
    body = body.replace(" -", " ")
    body = body.replace("- ", " ")
    return body


def remove_stopwords(word_list, black_list, white_list):
    """ Returns a list of words (with stop words removed), given a word list """
    stop = set(stopwords.words('english'))
    return [word for word in word_list
            if (word in white_list) or ((word not in stop) and (word not in black_list))]


def stem_words(word_list):
    """ Uses PorterStemmer to stem words in word_list argument """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in word_list]


def replace_number(word):
    """ Given a string, returns '&NUM' if it's a number and the input string otherwise """
    if word.isdigit():
        return '&NUM'

    return word


def group_numbers(word_list):
    """ Given a word list, returns the same word list with all numbers replaced with '&NUM' """
    return [replace_number(word) for word in word_list]


def merge(n_gram):
    ret = ''
    for word in n_gram:
        ret += '{} '.format(word)

    return ret[:-1]


def get_n_grams(word_list, gram_size):
    """ Given a word list and some gram size, returns a list of all n grams for n <= gram_size """
    if gram_size == 1:
        ret = word_list
    else:
        ret = [merge(el) for el in ngrams(word_list, gram_size)] + get_n_grams(word_list, gram_size - 1)

    return ret


def generate_word_list(body, settings):
    """ Returns a list of words, given a message tag """
    body = remove_punctuation(body)
    body = body.lower()

    word_list = body.split()

    if settings['remove_stopwords']:
        word_list = remove_stopwords(word_list, settings['black_list'], settings['white_list'])

    if settings['stem_words']:
        word_list = stem_words(word_list)

    if settings['group_numbers']:
        word_list = group_numbers(word_list)

    return get_n_grams(word_list, settings['n_gram'])


def normalise(feature):
    norm = linalg.norm(feature)
    if norm == 0:
        ret = feature
    else:
        ret = [el / norm for el in feature]

    return ret


def normalise_features(features):
    for feature in features:
        feature['speech_bag'] = normalise(feature['speech_bag'])

    return features


def condense_bag(feature, words):
    return [feature[word] for word in words]


def condense_bags(features, words):
    """ Returns an array of integer arrays containing the counts of the words
         (in the array provided) and an array of the Counter bags """

    for feature in features:
        feature['speech_bag'] = condense_bag(feature['speech_bag'], words)

    return features


def generate_classifier_data(aye_features, no_features, common_words, normalise_data):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the bags and most common words in them """

    features = aye_features + no_features
    features = condense_bags(features, common_words)
    if normalise_data:
        features = normalise_features(features)
    samples = []

    for _ in range(len(aye_features)):
        samples.append(1)

    for _ in range(len(no_features)):
        samples.append(-1)

    return features, samples


def compute_rank(s):
    min_val = s[0] / 100

    i = 0
    val = s[i]
    while val > min_val:
        i += 1
        val = s[i]

    return i - 1


def generate_linear_param_sets(linear_param_values):
    linear_param_sets = []
    for c_value in linear_param_values['cs']:
        linear_param_sets.append({
            'c': c_value
        })

    return linear_param_sets


def generate_rbf_param_sets(rbf_param_values):
    rbf_param_sets = []
    for c_value in rbf_param_values['cs']:
        for gamma_value in rbf_param_values['gammas']:
            rbf_param_sets.append({
                'c': c_value,
                'gamma': gamma_value
            })

    return rbf_param_sets


def generate_poly_param_sets(poly_param_values):
    poly_param_sets = []
    for c_value in poly_param_values['cs']:
        for gamma_value in poly_param_values['gammas']:
            for d_value in poly_param_values['ds']:
                for r_value in poly_param_values['rs']:
                    poly_param_sets.append({
                        'c': c_value,
                        'gamma': gamma_value,
                        'd': d_value,
                        'r': r_value
                    })

    return poly_param_sets


def generate_lower_log_params(max_param, no_of_params, log_diff):
    return logspace(log10(max_param) - (no_of_params - 1) * log10(log_diff), log10(max_param), no_of_params)


def generate_higher_log_params(min_param, no_of_params, log_diff):
    return logspace(log10(min_param), log10(min_param) + (no_of_params - 1) * log10(log_diff), no_of_params)


def generate_lower_params(max_param, no_of_params, diff):
    return [(diff * i) - (diff*(no_of_params - 1) - max_param) for i in range(no_of_params)]


def generate_higher_params(min_param, no_of_params, diff):
    return [(diff * i) + min_param for i in range(no_of_params)]


def compute_t(differences):
    avg = sum(differences) / len(differences)

    squared_diff = 0

    for difference in differences:
        squared_diff += pow(difference - avg, 2)

    variance = squared_diff / (len(differences) - 1)

    return avg / (math.sqrt(variance / len(differences)))


def generate_linear_values(no_of_cs):
    c_values = logspace(-3, 1, no_of_cs)

    return {
        'cs': c_values
    }


def generate_rbf_values(no_of_cs, no_of_gammas):
    c_values = logspace(-3, 1, no_of_cs)
    gamma_values = logspace(-3, 1, no_of_gammas)

    return {
        'cs': c_values,
        'gammas': gamma_values
    }


def generate_poly_values(no_of_cs, no_of_gammas, no_of_ds, no_of_rs):
    c_values = logspace(-3, 1, no_of_cs)
    gamma_values = logspace(-3, 3, no_of_gammas)
    d_values = [i for i in range(2, 2 + no_of_ds)]
    r_values = [i for i in range(0, no_of_rs)]

    return {
        'cs': c_values,
        'gammas': gamma_values,
        'ds': d_values,
        'rs': r_values
    }


def generate_refined_linear_values(linear_params, no_of_cs):

    c_values = logspace(log10(linear_params['c']) - 1, log10(linear_params['c']) + 1, no_of_cs)

    return {
        'cs': c_values
    }


def generate_refined_rbf_values(rbf_params, no_of_cs, no_of_gammas):

    c_values = logspace(log10(rbf_params['c']) - 1, log10(rbf_params['c']) + 1, no_of_cs)

    gamma_values = logspace(log10(rbf_params['gamma']) - 1, log10(rbf_params['gamma']) + 1, no_of_gammas)

    return {
        'cs': c_values,
        'gammas': gamma_values
    }


def generate_refined_poly_values(poly_params, no_of_cs, no_of_gammas, no_of_ds, no_of_rs):

    c_values = logspace(log10(poly_params['c']) - 1, log10(poly_params['c']) + 1, no_of_cs)

    gamma_values = logspace(log10(poly_params['gamma']) - 1, log10(poly_params['gamma']) + 1, no_of_gammas)

    d_values = [i for i in range(poly_params['d'], poly_params['d'] + no_of_ds)]

    r_values = [i for i in range(poly_params['r'], poly_params['r'] + no_of_rs)]

    return {
        'cs': c_values,
        'gammas': gamma_values,
        'ds': d_values,
        'rs': r_values
    }
