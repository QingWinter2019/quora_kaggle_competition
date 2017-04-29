'''
__file__

    ngram.py

__description__

    This file provides functions to compute n-gram & n-term.

__author__

    Corrected from Chenglong Chen < c.chenglong@gmail.com > original file

'''

import unittest


def get_bigram(words, join_string="_"):
    '''Return bigrams.

    Parameters:
    -----------
        words: iterable over str.
            List of words.

    Returns:
    --------
        bigrams: iterable over str.
            List of bigrams.
    '''

    l = len(words)
    if l > 1:
        bigrams = []
        for i in range(l - 1):
            bigrams.append(join_string.join([words[i], words[i+1]]))
    else:
        # Set it as unigram.
        bigrams = words

    return bigrams


class BigramTest(unittest.TestCase):

    def test_single_word(self):
        words = ['Love']
        self.assertEqual(get_bigram(words), ['Love'])

    def test_two_words(self):
        words = ['I', 'Love']
        self.assertEqual(get_bigram(words), ['I_Love'])

    def test_three_words(self):
        words = ['I', 'Love', 'you']
        self.assertEqual(get_bigram(words), ['I_Love', 'Love_you'])
