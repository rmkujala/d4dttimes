import aux

import unittest
from collections import Counter

import numpy as np


class Test(unittest.TestCase):

    def test_counter_to_array(self):
        a = [1, 2, 3, 4, 1, 2, 3, 3]
        c = Counter(a)
        ac = aux.counter_to_array(c)
        assert np.sum(ac == 1) == 2
        assert np.sum(ac == 2) == 2
        assert np.sum(ac == 3) == 3
        assert np.sum(ac == 4) == 1
