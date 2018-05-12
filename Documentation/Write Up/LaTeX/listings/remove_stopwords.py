def remove_stopwords(word_list, black_list, white_list):
    ''' Returns a list of words (with stop words removed), given a word list '''

    stop = set(stopwords.words('english'))
    return [word for word in word_list
            if (word in white_list) or ((word not in stop) and (word not in black_list))]