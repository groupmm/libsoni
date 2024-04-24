import numpy as np
import pandas as pd
from unittest import TestCase

from libsoni.core.spectrogram import sonify_spectrogram


class TestSpectrogram(TestCase):
    def setUp(self) -> None:
        self.fs = 22050
        self.spect = np.random.rand(10, 20)
        self.freq_coeff = np.random.rand(10)
        self.time_coeff = np.random.rand(20)

    def test_input_types(self) -> None:
        self.assertIsInstance(self.fs, int)
        self.assertIsInstance(self.spect, np.ndarray)

    def test_spect_shapes(self) -> None:
        with self.assertRaises(ValueError) as context:
            _ = sonify_spectrogram(spectrogram=self.spect,
                                   frequency_coefficients=self.freq_coeff[:-1],
                                   time_coefficients=self.time_coeff)

        self.assertEqual(str(context.exception), 'The length of frequency_coefficients must match spectrogram.shape[0]')

        with self.assertRaises(ValueError) as context:
            _ = sonify_spectrogram(spectrogram=self.spect,
                                   frequency_coefficients=self.freq_coeff,
                                   time_coefficients=self.time_coeff[:-1])

        self.assertEqual(str(context.exception), 'The length of time_coefficients must match spectrogram.shape[1]')



