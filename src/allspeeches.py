import collections

import database

from trainhelper import generate_word_list

def count_corpus():
    total = 0
    bag = collections.Counter()
    settings = {
        'n_gram': 1,
        'group_numbers': False,
        'remove_stopwords': False,
        'stem_words': False
    }
    with database.Database() as corpus:
        speeches = [speech[0] for speech in corpus.get_all_speeches()]
        print('{} speeches'.format(len(speeches)))
        for speech in speeches:
            word_list = generate_word_list(speech, settings)
            total += len(word_list)
            for word in word_list:
                bag[word] += 1

    print('{} tokens'.format(total))
    print('{} words'.format(len(bag)))


count_corpus()
