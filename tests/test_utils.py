import numpy as np
import soundfile as sf
from unittest import TestCase


from libsoni import utils


class TestF0(TestCase):
    def setUp(self) -> None:
        self.test_pitch = np.array([400,400,400,400,0,0,0,0,0,0,400,400,400,400,401,401,200,200,200])
        self.test_amps = np.ones(len(self.test_pitch))
        self.reference_array_split = [np.array([400,400,400,400]),np.array([0,0,0,0,0,0]),np.array([400,400,400,400,401,401]),np.array([200,200,200])]
        self.reference_replace = np.array([400,400,400,400,400,400,400,400,400,400,400,400,400,400,401,401,200,200,200])


    def test_split(self):
        splits = utils.split_freq_trajectory(frequencies = self.test_pitch, max_change_cents = 50)
        freq = np.split(self.test_pitch, splits)
        self.assertEqual(len(freq), len(self.reference_array_split), msg='number of splits does not match')

    def test_replace_zeros(self):

        test = utils.replace_zeros(self.test_pitch, 3)
        self.assertEqual(test[6], 0, msg='Too many zeros replaced')
        
        test = utils.replace_zeros(self.test_pitch, 7)
        self.assertEqual(test[6], self.reference_replace[6], msg='No zeros replaced')

        
        