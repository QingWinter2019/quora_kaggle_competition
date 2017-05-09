import logging
import pandas as pd
from nltk.corpus import wordnet as wn

from feature_classification import add_features, dump_feature_classes_and_dict
from pickle_utils import dump_features, check_if_exists


def synonyms(word1, word2):

    syns1 = set(wn.synsets(word1))
    syns2 = set(wn.synsets(word2))

    return bool(syns1.intersection(syns2))


def antonyms(word1, word2):

    # Antonym synsets for word1.
    syns1 = []
    for syn in wn.synsets(word1):
        for l in syn.lemmas():
            for a in l.antonyms():
                syns1.append(a.synset())
    syns1 = set(syns1)

    # Synonym synsets for word2.
    syns2 = set(wn.synsets(word2))

    return bool(syns1.intersection(syns2))


def hyponyms(word1, word2):

    hypo1 = []
    for syn in wn.synsets(word1):
        hypo1 += syn.hyponyms()
    hypo1 = set(hypo1)

    hypo2 = []
    for syn in wn.synsets(word1):
        hypo2 += syn.hyponyms()
    hypo2 = set(hypo2)

    return bool(hypo1.intersection(hypo2))


def hypernyms(word1, word2):

    hyper1 = []
    for syn in wn.synsets(word1):
        hyper1 += syn.hypernyms()
        break
    hyper1 = set(hyper1)

    hyper2 = []
    for syn in wn.synsets(word1):
        hyper2 += syn.hypernyms()
        break
    hyper2 = set(hyper2)

    return bool(hyper1.intersection(hyper2))


def synonyms_count(words1, words2):

    count = 0
    for word1 in words1:
        for word2 in words2:
            if word1 == word2:
                count += 1
                continue
            if synonyms(word1, word2):
                count += 1
                continue
    return count


def antonyms_count(words1, words2):

    count = 0
    for word1 in words1:
        for word2 in words2:
            if word1 == word2:
                continue
            if antonyms(word1, word2):
                count += 1
                continue
    return count


def hyponyms_count(words1, words2):

    count = 1
    for word1 in words1:
        for word2 in words2:
            if word1 == word2:
                count += 1
                continue
            if hyponyms(word1, word2):
                count += 1
                continue
    return count


def hypernyms_count(words1, words2):

    count = 1
    for word1 in words1:
        for word2 in words2:
            if word1 == word2:
                count += 1
                continue
            if hypernyms(word1, word2):
                count += 1
                continue
    return count


def words_len(words):
    return sum([len(word) for word in words])


def common_words_count(words1, words2):
    set1 = set(words1)
    set2 = set(words2)
    return len(set1.intersection(set2))


def common_words_len(words1, words2):
    set1 = set(words1)
    set2 = set(words2)
    return words_len(set1.intersection(set2))


def union_words_count(words1, words2):
    set1 = set(words1)
    set2 = set(words2)
    return len(set1.union(set2))


def union_words_len(words1, words2):
    set1 = set(words1)
    set2 = set(words2)
    return words_len(set1.union(set2))


def create_wordnet_features(data, pref=''):

    feature_class = pref + 'wordnet'
    logging.info('Creating wordnet (%s) features' % feature_class)
    if check_if_exists(feature_class):
        logging.info('Wordnet (%s) features already created' % feature_class)
        return

    res = pd.DataFrame()
    logging.info('Creating synonyms count...')
    res['synonyms_count'] = data.apply(
        lambda x: synonyms_count(x['words1'], x['words2']), axis=1)
    logging.info('Creating antonyms count...')
    res['antonyms_count'] = data.apply(
        lambda x: antonyms_count(x['words1'], x['words2']), axis=1)
#    logging.info('Creating hyponyms count...')
#    res['hyponyms_count'] = data.apply(
#        lambda x: hyponyms_count(x['words1'], x['words2']), axis=1)
#    logging.info('Creating hypernyms count...')
#    res['hypernyms_count'] = data.apply(
#        lambda x: hypernyms_count(x['words1'], x['words2']), axis=1)
    logging.info('Calculating synonyms and antonyms distances...')
    len1 = data['words1'].apply(lambda x: len(x))
    len2 = data['words2'].apply(lambda x: len(x))
    lenunion = data.apply(
        lambda x: union_words_count(x['words1'], x['words2']), axis=1)

    res['syn_distance1'] = res['synonyms_count'] / lenunion
    res['syn_distance2'] = res['synonyms_count'] / (len1 + len2)

    res['anton_distance1'] = res['antonyms_count'] / lenunion
    res['anton_distance2'] = res['antonyms_count'] / (len1 + len2)

    features = res.columns.tolist()
    add_features(feature_class, features)
    dump_features(feature_class, res)
    logging.info('Common words features are created and saved to pickle file.')
