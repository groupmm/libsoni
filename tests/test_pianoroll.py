import numpy as np
import pandas as pd
import soundfile as sf
from unittest import TestCase

from libsoni.core.pianoroll import sonify_pianoroll_clicks,\
    sonify_pianoroll_additive_synthesis, sonify_pianoroll_fm_synthesis, sonify_pianoroll_sample
from libsoni.util.utils import format_df, check_df_schema


class TestPianoRoll(TestCase):
    def setUp(self) -> None:
        self.df_pianoroll = pd.read_csv('data_csv/demo_pianoroll/FMP_B_Sonify_Pitch_Schubert_D911-11_SC06.csv', sep=';')
        self.fs = 22050
        self.sample, _ = sf.read('data_audio/samples/01Pia1F060f_np___0_short.wav', self.fs)
        self.durations = [int(3.0*self.fs), int(5.0*self.fs)]
        self.partials = [np.array([1]), np.array([1, 2, 3])]
        self.partials_amplitudes = [np.array([1]), np.array([1, 0.5, 0.25])]

    def test_input_types(self) -> None:
        [self.assertIsInstance(duration, int) for duration in self.durations]
        [self.assertIsInstance(partials, np.ndarray) for partials in self.partials]
        [self.assertIsInstance(partials_amplitude, np.ndarray) for partials_amplitude in self.partials_amplitudes]
        self.assertIsInstance(self.fs, int)
        check_df_schema(self.df_pianoroll)

    def test_input_df(self) -> None:
        incorrect_df = self.df_pianoroll.copy().rename(columns=str.lower)
        incorrect_df['tmp'] = None
        with self.assertRaises(ValueError) as context:
            _ = format_df(incorrect_df)

        self.assertEqual(str(context.exception), "Columns of the dataframe must be ['start', 'duration',"
                                                 " 'pitch', 'velocity', 'label'].")

    def test_pianoroll_additive(self) -> None:
        for partial_idx, partials in enumerate(self.partials):
            y = sonify_pianoroll_additive_synthesis(pianoroll_df=self.df_pianoroll,
                                                    partials=partials,
                                                    partials_amplitudes=self.partials_amplitudes[partial_idx],
                                                    sonification_duration=int(5*self.fs))

            ref, _ = sf.read(f'tests/data/pianoroll_add_{partial_idx}.wav')
            assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)

    def test_pianoroll_sample(self) -> None:
        y = sonify_pianoroll_sample(pianoroll_df=self.df_pianoroll,
                                    sample=self.sample,
                                    reference_pitch=60,
                                    sonification_duration=int(3.0*self.fs))
        ref, _ = sf.read(f'tests/data/pianoroll_sample.wav')
        assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)

    def test_pianoroll_sample(self) -> None:
        y = sonify_pianoroll_sample(pianoroll_df=self.df_pianoroll,
                                    sample=self.sample,
                                    reference_pitch=60,
                                    sonification_duration=int(3.0*self.fs))
        ref, _ = sf.read(f'tests/data/pianoroll_sample.wav')
        assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)

    def test_pianoroll_fm(self) -> None:
        y = sonify_pianoroll_fm_synthesis(pianoroll_df=self.df_pianoroll,
                                          sonification_duration=int(3.0 * self.fs),
                                          mod_rate_relative=0.5,
                                          mod_amp=0.1)
        ref, _ = sf.read(f'tests/data/pianoroll_fm.wav')
        assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)

    def test_pianoroll_clicks(self) -> None:
        y = sonify_pianoroll_clicks(pianoroll_df=self.df_pianoroll,
                                    sonification_duration=int(3.0 * self.fs))
        ref, _ = sf.read(f'tests/data/pianoroll_clicks.wav')
        assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)

