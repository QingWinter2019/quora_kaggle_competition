"""
__file__

    preprocessing.py

__description__

    Preprocess data in different ways.

__author__

    Sergii Golovko < sergii.golovko@gmail.com >

"""

import logging
import nltk
import os
import pandas as pd
import pickle
import re
from nltk.stem.snowball import PorterStemmer
from nltk.corpus import stopwords
from jellyfish import damerau_levenshtein_distance

from globals import CONFIG

# Global directories.
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(BASE_DIR, CONFIG['DATA_DIR'])
OUTPUT_DIR = os.path.join(BASE_DIR, CONFIG['OUTPUT_DIR'])
PREPROCESS_DIR = os.path.join(OUTPUT_DIR, CONFIG['PREPROCESS_DIR'])

# Global files.
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# Number of rows to read from files.
TEST_NROWS = CONFIG['TEST_NROWS']
TRAIN_NROWS = CONFIG['TRAIN_NROWS']
DL_COUNT = 0
CONCAT_COUNT = 0

# Stopwords.
stop_words = set(stopwords.words('english'))
stop_words.update(['possible', 'list'])


def save_preprocessed_data(data, name):

    if not os.path.exists(PREPROCESS_DIR):
        os.makedirs(PREPROCESS_DIR)

    path = os.path.join(PREPROCESS_DIR, name + '.pickle')
    with open(path, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def load_preprocessed_data(name):

    path = os.path.join(PREPROCESS_DIR, name + '.pickle')
    if not os.path.exists(path):
        raise ValueError('File %s does not exists.' % path)

    return pickle.load(open(path, 'rb'))


def check_if_preprocessed_data_exists(name):
    return os.path.exists(os.path.join(PREPROCESS_DIR, name + '.pickle'))


def create_words(str_, regex=r'\W+'):

    new_str = re.sub(regex, ' ', str_.lower())
    return new_str.strip().split(' ')


def create_sentence_words(sentences, regex=r'\W+'):

    words = []
    for sentence in sentences:
        end = ''
        if len(sentence) == 0:
            continue
        if sentence[-1] in ('.', '?', '!'):
            end = sentence[-1]
        words += create_words(sentence)
        if len(end) > 0:
            words.append(end)

    return words


def create_sentences(str_):

    sentences = nltk.sent_tokenize(str_)
    return sentences


def find_last_question(str_):

    try:
        sentences = nltk.sent_tokenize(str_)
    except:
        return ''
    for sentence in reversed(sentences):
        if sentence.endswith('?'):
            if len(sentences) > 1:
                logging.debug(sentence)
            return sentence
    # If no question is found.
    return ''


def dl_preprocess_words(words1, words2):
    global DL_COUNT
    min_distance = 100
    min_threshold = 0.4
    new_words = []
    for word1 in words1:
        l1 = len(word1)
        if l1 < 5:
            new_words.append(word1)
            continue
        closest_word = word1
        for word2 in words2:
            l2 = len(word2)
            if l2 < 5:
                continue
            try:
                d = damerau_levenshtein_distance(word1, word2)
            except:
                d = 100
            if d < min_threshold * min(l1, l2) and d < min_distance:
                min_distance = d
                closest_word = word2
                DL_COUNT += 1
                logging.debug('count: %d, word1: %s, word2: %s, distance: %d' %
                              (DL_COUNT, word1, word2, d))
        new_words.append(closest_word)
    return new_words


def concat_preprocess_words(words1, words2):
    '''Try to concatenate words in words1 to get word in words2.
    Looking for a first concatenation as for now.
    '''

    global CONCAT_COUNT
    prev_concat_count = CONCAT_COUNT

    new_words, i, len1 = [], 0, len(words1)
    while i < len1:
        word1 = words1[i]
        # The word(s) in words1 that will be substituted.
        substitute_word = word1
        for word2 in words2:
            # Do not need to concatenate if words are the same.
            if word1 == word2:
                break
            if word1 in word2:
                j = i + 1
                concat_word = word1
                found = False
                while j < len1:
                    concat_word += words1[j]
                    # Found write way to concatenate.
                    if concat_word == word2:
                        substitute_word = word2
                        CONCAT_COUNT += 1
                        concat_words = [word for word in words1[i:j+1]]
                        logging.debug('Count: %d, substitute word: %s, concatenated words: %s'
                                      % (CONCAT_COUNT, substitute_word, concat_words))
                        i = j
                        found = True
                        break
                    # Need to look whether concatenating one more word will help.
                    elif concat_word in word2:
                        j += 1
                    # Impossible to concatenate to word2.
                    else:
                        break
                if found:
                    break
        new_words.append(substitute_word)
        i += 1

    if prev_concat_count == CONCAT_COUNT:
        assert words1 == new_words, ('Nothing concatenated, words1 and new '
                                     'words must be the same.')

    return new_words


def subset_pos_taggers(txt, tagset=['NN', 'NNP', 'NNPS', 'NNS']):
    words_tokenized = nltk.word_tokenize(txt)
    words_pos_tagged = nltk.pos_tag(words_tokenized)
    is_in_tagset = lambda pos: pos[:2] in tagset
    new_words = [word for (word, pos) in words_pos_tagged if is_in_tagset(pos)]
    return new_words


def preprocess_data():

    logging.info('PREPROCESSING DATA')

    # Read data.
    df_train = pd.read_csv(TRAIN_FILE, nrows=TRAIN_NROWS)
    df_test = pd.read_csv(TEST_FILE, nrows=TEST_NROWS)
    df_test.rename(columns={'test_id': 'id'}, inplace=True)

    # Merge data together.
    wanted_cols = ['id', 'question1', 'question2']
    data = pd.concat([df_train[wanted_cols + ['is_duplicate']],
                      df_test[wanted_cols]], ignore_index=True)

    # Create standard preprocessing: split data by non-alphanumerical ch,
    # lower case.
    name = 'standard_preprocess'
    if not check_if_preprocessed_data_exists(name):
        data_preprocessed = pd.DataFrame(data['id'])
        data_preprocessed['question1'] = data['question1'].apply(
            lambda x: str(x))
        data_preprocessed['question2'] = data['question2'].apply(
            lambda x: str(x))
        data_preprocessed['words1'] = data_preprocessed['question1'].apply(
            lambda x: create_words(x))
        data_preprocessed['words2'] = data_preprocessed['question2'].apply(
            lambda x: create_words(x))
        save_preprocessed_data(data_preprocessed, name)

    name = 'stemma_preprocess'
    if not check_if_preprocessed_data_exists(name):
        # Load standard preprocessed data.
        data_preprocessed = load_preprocessed_data('standard_preprocess')

        # Stemmatize words.
        stemmer = PorterStemmer(ignore_stopwords=False)
        data_preprocessed['words1'] = data_preprocessed['words1'].apply(
            lambda x: [stemmer.stem(word) for word in x])
        data_preprocessed['words2'] = data_preprocessed['words2'].apply(
            lambda x: [stemmer.stem(word) for word in x])
        data_preprocessed['question1'] = data_preprocessed['words1'].apply(
            lambda x: ' '.join(x))
        data_preprocessed['question2'] = data_preprocessed['words2'].apply(
            lambda x: ' '.join(x))
        save_preprocessed_data(data_preprocessed, name)

    name = 'stemma_preprocess_stopwords'
    if not check_if_preprocessed_data_exists(name):
        # Load standard preprocessed data.
        data_preprocessed = load_preprocessed_data('standard_preprocess')

        # Stemmatize words.
        stemmer = PorterStemmer(ignore_stopwords=False)
        data_preprocessed['words1'] = data_preprocessed['words1'].apply(
            lambda x: [word for word in x if word not in stop_words])
        data_preprocessed['words2'] = data_preprocessed['words2'].apply(
            lambda x: [word for word in x if word not in stop_words])
        data_preprocessed['words1'] = data_preprocessed['words1'].apply(
            lambda x: [stemmer.stem(word) for word in x])
        data_preprocessed['words2'] = data_preprocessed['words2'].apply(
            lambda x: [stemmer.stem(word) for word in x])
        data_preprocessed['question1'] = data_preprocessed['words1'].apply(
            lambda x: ' '.join(x))
        data_preprocessed['question2'] = data_preprocessed['words2'].apply(
            lambda x: ' '.join(x))
        save_preprocessed_data(data_preprocessed, name)

    name = 'dl_preprocess'
    if not check_if_preprocessed_data_exists(name):
        logging.info('Doing Damerau Levenstein preprocessing.')
        # Load standard preprocessed data.
        data_preprocessed = load_preprocessed_data('stemma_preprocess_stopwords')

        # Stemmatize words.
        data_preprocessed['words1'] = data_preprocessed.apply(
            lambda x: dl_preprocess_words(x['words1'], x['words2']), axis=1)
        data_preprocessed['question1'] = data_preprocessed['words1'].apply(
            lambda x: ' '.join(x))
        save_preprocessed_data(data_preprocessed, name)

    name = 'concat_preprocess'
    if not check_if_preprocessed_data_exists(name):
        logging.info('Doing concat preprocessing.')
        data_preprocessed = pd.DataFrame(data['id'])
        data_preprocessed['question1'] = data['question1'].apply(
            lambda x: str(x))
        data_preprocessed['question2'] = data['question2'].apply(
            lambda x: str(x))
        data_preprocessed['words1'] = data_preprocessed['question1'].apply(
            lambda x: create_words(x))
        data_preprocessed['words2'] = data_preprocessed['question2'].apply(
            lambda x: create_words(x))
        data_preprocessed['words1'] = data_preprocessed.apply(
            lambda x: concat_preprocess_words(x['words1'], x['words2']), axis=1)
        data_preprocessed['words2'] = data_preprocessed.apply(
            lambda x: concat_preprocess_words(x['words2'], x['words1']), axis=1)

        save_preprocessed_data(data_preprocessed, name)

    name = 'noun_preprocess'
    if not check_if_preprocessed_data_exists(name):
        logging.info('Doing noun preprocessing.')
        data_preprocessed = pd.DataFrame(data['id'])
        data_preprocessed['question1'] = data['question1'].apply(
            lambda x: str(x))
        data_preprocessed['question2'] = data['question2'].apply(
            lambda x: str(x))
        data_preprocessed['words1'] = data_preprocessed['question1'].apply(
            lambda x: create_sentence_words(create_sentences(x)))
        data_preprocessed['words2'] = data_preprocessed['question2'].apply(
            lambda x: create_sentence_words(create_sentences(x)))
        data_preprocessed['words1'] = data_preprocessed.apply(
            lambda x: concat_preprocess_words(x['words1'], x['words2']), axis=1)
        data_preprocessed['words2'] = data_preprocessed.apply(
            lambda x: concat_preprocess_words(x['words2'], x['words1']), axis=1)
        data_preprocessed['question1'] = data_preprocessed['words1'].apply(
            lambda x: ' '.join(x))
        data_preprocessed['question2'] = data_preprocessed['words2'].apply(
            lambda x: ' '.join(x))
        data_preprocessed['words1'] = data_preprocessed['question1'].apply(
            lambda x: subset_pos_taggers(x))
        data_preprocessed['words2'] = data_preprocessed['question2'].apply(
            lambda x: subset_pos_taggers(x))

        save_preprocessed_data(data_preprocessed, name)

    name = 'last_question_preprocess'
    if not check_if_preprocessed_data_exists(name):
        logging.info('Doing last question preprocessing.')
        data_preprocessed = pd.DataFrame(data['id'])
        data_preprocessed['question1'] = data['question1'].apply(
            lambda x: str(x))
        data_preprocessed['question2'] = data['question2'].apply(
            lambda x: str(x))
        data_preprocessed['question1'] = data['question1'].apply(
            lambda x: find_last_question(x))
        data_preprocessed['question2'] = data['question2'].apply(
            lambda x: find_last_question(x))

    logging.info('DATA PREPROCESSED')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    preprocess_data()
