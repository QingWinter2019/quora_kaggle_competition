from math import *
from collections import Counter

max_freqs = {}
idf_scores = Counter()


def create_idf(corpus):

    global idf_scores
    for s in corpus:
        idf_scores.update(set(str(s).split()))

    l = len(corpus)
    for s in idf_scores:
        idf_scores[s] = log(l/idf_scores[s])


def tfidf_score(word, str1, type='binary'):

    global idf_scores
    global max_freqs
    K = 0
    words = str1.split()

    if type == 'all':
        scores = []

        # binary score
        scores.append(idf_scores[word])

        # freq score
        freq = words.count(word)
        scores.append(freq * idf_scores[word])

        # log freq score
        scores.append(log(1 + freq) * idf_scores[word])

        # compute max freq
        max_freq = 0
        if str1 in max_freqs:
            max_freq = max_freqs[str1]
        else:
            for word in words:
                freq1 = words.count(word)
                max_freq = max(max_freq, freq1)
            max_freqs[str1] = max_freq

        # dnorm score
        scores.append((K + (1-K) * freq/max_freq) * idf_scores[word])

        return scores

    if type == 'binary':
        return idf_scores[word]

    # compute freq
    freq = words.count(word)

    if type == 'freq':
        return freq * idf_scores[word]

    if type == 'log_freq':
        return log(1 + freq) * idf_scores[word]

    # compute max freq
    max_freq = 0
    if str1 in max_freqs:
        max_freq = max_freqs[str1]
    else:
        for word in words:
            freq1 = words.count(word)
            max_freq = max(max_freq, freq1)
        max_freqs[str1] = max_freq

    if type == 'dnorm':
        return (K + (1-K) * freq/max_freq) * idf_scores[word]


# transform string of strings of length 4 to string of 4 strings
def transform_scores(scores):

    return_scores = []
    for j in range(4):
        return_scores.append([])
        for i in range(len(scores)):
            return_scores[j].append(scores[i][j])

    return return_scores


def mean_scores(scores):

    return_scores = []
    for i in range(len(scores)):
        return_scores.append(sum(scores[i])/len(scores[i]))

    return return_scores


def factorize(score, factor):

    if type(score) == list:
        return [ind_score * factor for ind_score in score]
    else:
        return score * factor


def tfidf1(str1, str2, type='binary'):

    words1, words2 = str1.split(), str2.split()
    if type == 'all':
        scores = [[0, 0, 0, 0]]
    else:
        scores = [0]

    for word in words1:
        if word in words2:
            score = tfidf_score(word, str2, type=type)
            scores.append(score)

    if type == 'all':
        scores = transform_scores(scores)

        return sum(scores[0]), sum(scores[1]), sum(scores[2]), sum(scores[3]), \
               max(scores[0]), max(scores[1]), max(scores[2]), max(scores[3]), \
               sum(scores[0])/len(words1), sum(scores[1])/len(words1), \
               sum(scores[2])/len(words1), sum(scores[3])/len(words1)
    else:
        return sum(scores), max(scores), sum(scores) / len(words1)


def tfidf2(str1, str2, type='binary'):

    words1, words2 = str1.split(), str2.split()

    if type == 'all':
        scores = [[0, 0, 0, 0]]
    else:
        scores = [0]

    for word1 in words1:
        loc_scores = []
        found = False
        for word2 in words2:
            # look for word1 in word2
            if word1 in word2:
                found = True
                score = tfidf_score(word2, str2, type=type)
                factor = len(word1)/len(word2)
                loc_scores.append(factorize(score, factor))

            # look for word2 in word1
            if (len(word2) > 3) and (word2 in word1):
                found = True
                score = tfidf_score(word2, str2, type=type)
                factor = len(word2)/len(word1)
                loc_scores.append(factorize(score, factor))
        if found:
            if type == 'all':
                score = mean_scores(transform_scores(loc_scores))
            else:
                score = sum(loc_scores)/len(loc_scores)
            scores.append(score)

    if type == 'all':

        scores = transform_scores(scores)
        return sum(scores[0]), sum(scores[1]), sum(scores[2]), sum(scores[3]), \
               max(scores[0]), max(scores[1]), max(scores[2]), max(scores[3]), \
               sum(scores[0])/len(words1), sum(scores[1])/len(words1), \
               sum(scores[2])/len(words1), sum(scores[3])/len(words1)
    else:
        return sum(scores), max(scores), sum(scores) / len(words1)
