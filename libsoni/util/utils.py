import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

import os
from functools import cmp_to_key
import music21 as m21
import warnings

SAMPLES = ['bass-drum', 'click', 'hi-hat']


def mix_sonification_and_original(sonification: np.ndarray,
                                  original_audio: np.ndarray,
                                  gain_lin_sonification: float = 1.0,
                                  gain_lin_original_audio: float = 1.0):
    """
        This function takes a sonification and an original_audio and mixes it to stereo

        Parameters
        ----------
        sonification: np.ndarray
            sonification
        original_audio: np.ndarray
            original_audio
        gain_lin_sonification: float, default = 1.0
            linear gain for sonification
        gain_lin_original_audio: float, default = 1.0
            linear gain for original audio

        Returns
        -------
        stereo_audio : np.ndarray
            stereo mix of the signals
        """
    min_length = min(len(original_audio), len(sonification))

    original_audio = original_audio[:min_length]
    sonification = sonification[:min_length]

    # Calculate the RMS amplitude of each signal
    rms_signal1 = np.sqrt(np.mean(np.square(original_audio)))
    rms_signal2 = np.sqrt(np.mean(np.square(sonification)))

    # Normalize the signals to have the same RMS amplitude
    normalized_signal1 = original_audio * (rms_signal2 / rms_signal1)
    normalized_signal2 = sonification * (rms_signal1 / rms_signal2)

    stereo_audio = np.column_stack((gain_lin_original_audio * normalized_signal1, gain_lin_sonification * normalized_signal2)).T

    return stereo_audio


def load_sample(name: str = 'click', target_sampling_rate: int = 44100) -> np.ndarray:
    assert name in SAMPLES, 'No valid sample'
    original_sampling_rate, data = wavfile.read(os.path.join('../..', 'data_audio', 'samples', name + '.wav'))

    # Calculate the resampling ratio
    resampling_ratio = target_sampling_rate / original_sampling_rate

    # Resample the data using scipy's resample function
    resampled_data = resample(data, int(len(data) * resampling_ratio))

    return resampled_data, target_sampling_rate


def click(pitch: int = 69, amplitude: float = 1.0, duration: float = 0.2, fs: int = 44100,
          tuning_frequency: int = 440) -> np.ndarray:
    """
    Returns a click signal
    Parameters
    ----------
    pitch : int
        pitch for colored click
    amplitude : float
        amplitude of click signal
    duration : float
        duration of click signal
    fs : int
        sampling rate
    tuning_frequency : int
        tuning frequency
    Returns
    -------
    click : array-like
        click signal

    """
    click_freq = tuning_frequency * 2 ** ((pitch - 69) / 12)

    angular_freq = 2 * np.pi * click_freq / float(fs)

    click = np.logspace(0, -10, num=int(fs * duration), base=2.0)

    click *= np.sin(angular_freq * np.arange(len(click)))

    click *= amplitude

    return click


def generate_shepard_tone(pitch_class: int = 0,  # TODO: check sigma parameter
                          filter: bool = False,
                          f_center: float = 440.0,
                          octave_cutoff: int = 1,
                          amplitude: float = 1.0,
                          duration: float = 1.0,
                          fs: int = 44100,
                          f_tuning: float = 440,
                          fade_dur: float = 0.01,
                          ) -> np.ndarray:
    """Generate shepard tone

        Args:
            pitch_class: int (default: 0)
                pitch class of the synthesized tone
            filter: bool (default: False)
                decides, if shepard tones are filtered or not
            f_center : float (default: 440.0)
                center_frequency in Hertz for bell-shaped filter
            octave_cutoff: int (default: 1)
                determines, at which multiple of f_center, the harmonics get attenuated by 2.
            amplitude: float (default: 1.0)
                amplitude of resulting signal
            duration: float (default: 1.0)
                duration (in seconds)
            fs: int (default: 44100)
                sampling rate in Samples/second
            f_tuning: float (default: 440.0)
                tuning frequency (in Hz)
            fade_dur: float (default: 0.01)
                duration (in seconds) of fade in and fade out (to avoid clicks)

        Returns:
            y: synthesized tone
    """
    assert 0 <= pitch_class <= 11, "pitch class out of range"

    N = int(duration * fs)
    t = np.arange(N) / fs
    freqs = f_tuning * 2 ** ((pitch_class + np.arange(11) * 12 - 69) / 12)
    y = np.zeros(N)

    if duration < fade_dur * 2:
        return y
    if filter:
        f_log = 2 * np.logspace(1, 4, 20000)
        f_lin = np.linspace(20, 20000, 20000)
        f_center_lin = np.argmin(np.abs(f_log - f_center))
        weights = np.exp(- (f_lin - f_center_lin) ** 2 / (1.4427 * ((octave_cutoff * 2) * 1000) ** 2))

        for freq in freqs:
            y += weights[np.argmin(np.abs(f_log - freq))] * np.sin(2 * np.pi * freq * t)

    else:
        for freq in freqs:
            y += np.sin(2 * np.pi * freq * t)

    fade_samples = int(fade_dur * fs)

    y[0:fade_samples] *= np.linspace(0, 1, fade_samples)
    y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    y = amplitude * (y / np.max(y))
    return y


def add_to_sonification(sonification: np.ndarray, sonification_for_event: np.ndarray, start: float, fs: int = 44100):
    # TODO: recheck that boarders of sonification are kept
    """
    This function inserts the signal of a sonified event into the sonification array using the start time of the event.

    Parameters
    ----------
    sonification : np.ndarray
        sonification array
    sonification_for_event : np.ndarray
        signal to insert
    start : float
        start time for signal to insert
    fs : int
        sampling rate

    Returns
    ----------
    sonification: array-like
        sonification array with inserted signal
    """
    if (int(start * fs) + len(sonification_for_event)) < len(sonification):
        sonification[int(start * fs):(int(start * fs) + len(sonification_for_event))] += sonification_for_event
    return sonification


def generate_additive_synthesized_tone(pitch=69, frequency_ratios=[], frequency_ratios_amp=[], frequency_ratios_phase_offsets=[], amp=1, dur=1, fs=44100, f_tuning=440, fade_dur=0.01):
    """Generate additive synthesized tone

    Args:
        pitch: pitch of the synthesized tone
        frequency_ratios: overtone frequencies based on pitch frequency
        frequency_ratios_amp: amplitudes for overtone frequencies
        frequency_ratios_phase_offsets: phase offsets for overtone frequencies
        amp: amplitude of resulting signal
        dur: duration (in seconds)
        Fs: Sampling rate
        f_tuning: Tuning frequency
        fade_dur: Duration of fade in and fade out (to avoid clicks)

    Returns:
        y: synthesized tone
        t: time axis (in seconds)
    """
    N = int(dur * fs)
    t = np.arange(N) / fs
    y = np.zeros(len(t))
    if 2 * int(fade_dur * fs) > N:
        return y
    for i, frequency_ratio in enumerate(frequency_ratios):
        freq = f_tuning * 2 ** ((pitch - 69) / 12)
        y += frequency_ratios_amp[i] * np.sin(2 * np.pi * freq * frequency_ratio * t + frequency_ratios_phase_offsets[i])

    if not fade_dur == 0:
        fade_samples = int(fade_dur * fs)
        y[0:fade_samples] *= np.linspace(0, 1, fade_samples)
        y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    y = amp * (y / np.max(y))
    return y


def generate_fm_synthesized_tone(pitch=69, modulation_frequency=0, modulation_index=0, amp=1, dur=1, fs=44100, f_tuning=440, fade_dur=0.01):
    """Generate fm synthesized tone

    Args:
        pitch: pitch of the synthesized tone
        modulation_frequency: frequency to modulate
        modulation_index: strength of modulation
        amp: amplitude of resulting signal
        dur: duration (in seconds)
        Fs: Sampling rate
        f_tuning: Tuning frequency
        fade_dur: Duration of fade in and fade out (to avoid clicks)

    Returns:
        y: synthesized tone
        t: time axis (in seconds)
    """
    N = int(dur * fs)
    t = np.arange(N) / fs
    freq = f_tuning * 2 ** ((pitch - 69) / 12)
    y = np.sin(2 * np.pi * freq * t + modulation_index * np.sin(2 * np.pi * modulation_frequency * t))
    if not fade_dur == 0:
        fade_samples = int(fade_dur * fs)
        y[0:fade_samples] *= np.linspace(0, 1, fade_samples)
        y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    y = amp * (y / np.max(y))
    return y


def generate_sinusoid(frequency: float = 440.0,
                      amp: float = 1.0,
                      dur: float = 1.0,
                      fs: int = 44100,
                      fade_dur: float = 0.01) -> np.ndarray:
    """
    This function generates a sinusoid.

    Parameters
    ----------
        frequency : float, default = 440.0 Hz
            Frequency in Hertz for sinusoid
        amp : float, default = 1.0
            Amplitude for sinusoid
        dur : float, default = 1.0 s
            Duration in seconds for sinusoid
        fade_dur : float, default = 0.01 s
            Duration in seconds of fade in and fade out
        fs : int, default = 44100
            Sampling rate
    Returns
    -------
        y : np.ndarray
        sinusoid
    """
    N = int(dur * fs)
    t = np.arange(N) / fs

    y = np.sin(2 * np.pi * frequency * t)
    if not fade_dur == 0:
        fade_samples = int(fade_dur * fs)
        y[0:fade_samples] *= np.linspace(0, 1, fade_samples)
        y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    y = amp * (y / np.max(y))
    return y


def envelop_signal(signal: np.ndarray, attack_time: float = 0, decay_time: float = 0, sustain_level: float = 0, release_time: float = 0, fs=44100):
    """
    Envelopes a given signal. If the length of the signal is too short regarding the specified ADSR parameters, the returned signal is zero.
    Parameters
    ----------
    signal : array-like
        signal to envelope
    Returns
    ----------
    enveloped_signal: array-like
        enveloped signal
    """
    if attack_time <= 0 or decay_time <= 0 or release_time <= 0:
        return np.zeros(len(signal))

    # compute lengths of attack, decay, sustain and release section
    attack_samples = int(np.floor(attack_time * fs))
    decay_samples = int(np.floor(decay_time * fs))
    release_samples = int(np.floor(release_time * fs))
    sustain_samples = int(len(signal) - (attack_samples + decay_samples + release_samples))

    # check if signal is at least as long as attack, decay and release section
    if len(signal) < (attack_samples + decay_samples + release_samples):
        return np.zeros(len(signal))

    # compute attack section of envelope
    attack_func = np.exp(np.linspace(0, 1, int(np.floor(attack_time * fs)))) - 1
    attack_func = attack_func / np.max(np.flip(attack_func))

    # compute decay section of envelope
    decay_func = np.exp(np.linspace(0, 1, decay_samples)) - 1
    decay_func = np.flip(sustain_level + (1 - sustain_level) * (decay_func / np.max(decay_func)))

    # compute sustain section of envelope
    sustain_func = sustain_level * np.ones(sustain_samples)

    # compute release section of envelope
    release_func = np.exp(np.linspace(0, 1, release_samples)) - 1
    release_func = np.flip(sustain_level * (release_func / np.max(release_func)))

    # concatenate sections and envelope signal
    enveloped_signal = signal * np.concatenate([attack_func, decay_func, sustain_func, release_func])

    return enveloped_signal


""" Harte Chord Class -
    after Symbolic Representation of Musical Chords: a proposed syntax for text #annotations 2005 ISMIR

"""

_naturals = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
_modifiers = ['b', '#']
_intervals = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
_shorthands = {
    '9': ['3', '5', 'b7', '9'],
    'maj9': ['3', '5', '7', '9'],
    'min9': ['b3', '5', 'b7', '9'],
    'maj7': ['3', '5', '7'],
    'min7': ['b3', '5', 'b7'],
    '7': ['3', '5', 'b7'],
    'dim7': ['b3', 'b5', 'bb7'],
    'hdim7': ['b3', 'b5', 'b7'],
    'minmaj7': ['b3', '5', '7'],
    'maj6': ['3', '5', '6'],
    'min6': ['b3', '5', '6'],
    'maj': ['3', '5'],
    'min': ['b3', '5'],
    'dim': ['b3', 'b5'],
    'aug': ['3', '#5'],
    'sus4': ['4', '5']
}


class Hchord:
    def __init__(self, input_string=None, **kwargs):
        self._degrees = set()

        keywords = ['root', 'shorthand', 'degrees', 'bass']
        for val in kwargs:
            if val not in keywords:
                raise KeyError('Keyword ' + str(val) + ' does not exist!')

        root = kwargs.get(keywords[0], None)
        shorthand = kwargs.get(keywords[1], None)
        degrees = kwargs.get(keywords[2], None)
        bass = kwargs.get(keywords[3], None)

        if input_string is not None:
            if len(kwargs) != 0:
                raise ValueError('String input and keyword input not possible simultaneously')

            input_string = input_string.replace(' ', '')
            qua = None

            if len(input_string.split('/')) == 1:
                root_dgr = input_string
            elif len(input_string.split('/')) == 2:
                root_dgr = input_string.split('/')[0]
                bass = input_string.split('/')[1]
            else:
                raise ValueError('Input string is not in Harte Format')

            if len(root_dgr.split(':')) == 1:
                root = root_dgr
                qua = 'maj'
            elif len(root_dgr.split(':')) == 2:
                root = root_dgr.split(':')[0]
                qua = root_dgr.split(':')[1]
            else:
                raise ValueError('Input string is not in Harte Format')

            if len(qua.split('(')) == 1:
                shorthand = qua
            elif len(qua.split('(')) == 2:
                shorthand = qua.split('(')[0]
                if shorthand == '':
                    shorthand = None
                degrees = qua.split('(')[1][:-1]
            else:
                raise ValueError('Input string is not in Harte Format')

        # add root
        if root is None:
            self._root = None
        else:
            self.root(root)

        # add shorthand
        if shorthand is not None:
            self._add_shorthand(shorthand)

        # add degrees
        if degrees is not None:
            self.add_degree(degrees)

        # add bass degree
        if bass is None:
            self._bass_degree = '1'
        else:
            self.bass_degree(bass)

    def root(self, new_root=None):
        if new_root is None:
            return self._root
        else:
            if self._is_note(new_root):
                self._root = new_root
            elif '-' in new_root:
                self.root(new_root.replace('-', 'b'))
            else:
                raise TypeError('root has to be a valid combination out of A, B, C, D, E, F, G and #,b')

    def get_degrees(self):
        return self._degrees

    def bass_degree(self, new_bass=None):
        if new_bass is None:
            return self._root
        else:
            new_bass = new_bass.replace('-', 'b')
            if self._is_degree(new_bass) and (new_bass in self._degrees):
                self._bass_degree = new_bass
            elif self._is_note(new_bass):
                m21_bass = m21.pitch.Pitch(new_bass.replace('b', '-'))
                m21_root = m21.pitch.Pitch(self.root().replace('b', '-'))
                m21_int = m21.interval.Interval(noteStart=m21_root, noteEnd=m21_bass)
                if '-' in m21_int.directedName:
                    m21_int = m21_int.complement
                m21_int = m21.interval.Interval(m21_int.name)
                new_bass = _m21_interval_to_degree(m21_int)

                if (new_bass in self._degrees):
                    self.bass_degree(new_bass)
                else:
                    no = new_bass[-1]
                    new_bass = new_bass[:-1] + str(int(no) + 7)
                    self.bass_degree(new_bass)
            else:
                raise TypeError(
                    'bass degree has to be a valid combination out of 1,2,3,4,5,6,7,8,9,10,11,12,13 and #,b or a note Name and has to '
                    'be element of the chord')

    def add_degree(self, new_degree):
        if isinstance(new_degree, list):
            for val in new_degree:
                self.add_degree(val)
            return

        new_degree = new_degree.replace(' ', '')
        if new_degree[0] == '*':
            self.remove_degree(new_degree[1:])
        elif self._is_degree(new_degree):

            # check if generic degree already existent
            generic_degree = new_degree.replace('b', '')
            generic_degree = generic_degree.replace('#', '')
            for val in self._degrees:
                if (generic_degree in val) and (new_degree != val):
                    err = 'base degree already used! Added ' + new_degree + ', while ' + val + ' already existent.'
                    warnings.warn(err)

            self._degrees.add(new_degree)
        elif ',' in new_degree:
            self.add_degree(new_degree.split(','))
        else:
            raise TypeError(
                'degree has to be a valid combination out of 1,2,3,4,5,6,7,8,9,10,11,12,13 and #,b '
                'or a list of degrees')

    def clear_degrees(self):
        self._degrees.clear()

    def remove_degree(self, del_degree):
        if isinstance(del_degree, list):
            for val in del_degree:
                self.remove_degree(val)
        elif self._is_degree(del_degree):
            self._degrees.remove(del_degree)
        else:
            raise TypeError(
                'degree has to be a valid combination out of 1,2,3,4,5,6,7,8,9,10,11,12,13 and #,b '
                'or a list of degrees')

    def export_string(self, style='extended', absolute_bass=False, acc=None):
        if self.root() is None:
            return None

        if acc is not None:
            temp = Hchord(self.export_string())
            temp.root(self._get_enharmonic_name(acc))
            return temp.export_string(style=style, absolute_bass=absolute_bass)

        bass_string = ''
        if absolute_bass:
            if self._bass_degree != '1':
                m21_root = m21.pitch.Pitch(self._root.replace('b', '-'))
                m21_int = _degree_to_m21_interval(self._bass_degree)
                m21_bass = m21_int.transposePitch(m21_root)
                bass_string = '/' + str(m21_bass).replace('-', 'b')
        else:
            if self._bass_degree != '1':
                bass_string = '/' + self._bass_degree

        if style == 'shorthand':
            for _shorthand in _shorthands.keys():
                if set(_shorthands[_shorthand]) == self._degrees:
                    return (self._root + ':' + _shorthand + bass_string)
                elif set(_shorthands[_shorthand]).issubset(self._degrees):
                    degree_list = list(self._degrees.difference(set(_shorthands[_shorthand])))
                    separator = ', '
                    return (self._root + ':' + _shorthand + '(' + separator.join(
                        sorted(degree_list, key=cmp_to_key(_compare_degree))) + ')' + bass_string)
                else:
                    continue
            return self.export_string(style='extended', absolute_bass=absolute_bass)

        if style == 'extended':
            degree_list = list(self._degrees)
            separator = ', '
            return (self._root + ':(' + separator.join(
                sorted(degree_list, key=cmp_to_key(_compare_degree))) + ')' + bass_string)

        if style == 'majmin':
            return self.export_string(style='majminInv', absolute_bass=absolute_bass).split('/')[0]

        if style == 'majminInv':
            if 'b3' in self._degrees and '3' not in self._degrees:
                return (self._root + ':' + 'min' + bass_string)
            elif '3' in self._degrees and 'b3' not in self._degrees:
                return (self._root + ':' + 'maj' + bass_string)
            else:
                return (self._root + ':' + 'None' + bass_string)

        if style == 'majminNumeric':
            m21_root = m21.pitch.Pitch(self._root.replace('b', '-'))
            p_C = m21_root.pitchClass + 1
            quality = self.export_string(style='majmin', absolute_bass=absolute_bass).split(':')[1]
            if quality == 'maj':
                return int(p_C)
            elif quality == 'min':
                return int(p_C) + 12
            else:
                return 0

        # wrong style
        raise ValueError('style not valid')

    def export_chroma_vectors(self):
        root_arr = np.zeros(12, dtype=int)
        bass_arr = np.zeros(12, dtype=int)
        pitches_arr = np.zeros(12, dtype=int)

        m21_root = m21.pitch.Pitch(self._root.replace('b', '-'))
        root_class = int(m21_root.pitchClass)
        root_arr[root_class] = 1

        bass_arr[(root_class + _degree_to_halfsteps(self._bass_degree)) % 12] = 1

        for dgr in self._degrees:
            pitches_arr[(root_class + _degree_to_halfsteps(dgr)) % 12] = 1
        pitches_arr[root_class] = 1

        return root_arr, bass_arr, pitches_arr

    def transpose(self, interval):
        if self._is_degree(interval):
            m21_root = m21.pitch.Pitch(self._root.replace('b', '-'))
            m21_int = _degree_to_m21_interval(interval)

        else:
            TypeError('interval has to be a valid combination out of 1,2,3,4,5,6,7,8,9,10,11,12,13 and #,b')

    def _is_note(self, candidate, mod=None):

        if candidate in _naturals:
            return True
        else:
            if candidate[-1] in _modifiers and (mod is None or mod == candidate[-1]):
                return self._is_note(candidate[:-1], candidate[-1])
            else:
                return False

    def _is_degree(self, candidate, mod=None):

        if candidate in _intervals:
            return True
        else:
            if candidate[0] in _modifiers and (mod is None or mod == candidate[0]):
                return self._is_degree(candidate[1:], candidate[0])
            else:
                return False

    def _add_shorthand(self, shorthand):
        if shorthand in _shorthands.keys():
            self.add_degree(_shorthands[shorthand])
        else:
            raise TypeError(str(shorthand) + ' is no valid shorthand')

    def _get_enharmonic_name(self, accidental):
        m21_root = m21.pitch.Pitch(self._root.replace('b', '-'))
        m21_root.simplifyEnharmonic(inPlace=True)

        if accidental == 'sharp':
            while int(m21_root.alter) < 0:
                m21_root.getLowerEnharmonic(inPlace=True)
            return m21_root.name.replace('-', 'b')
        elif accidental == 'flat':
            while int(m21_root.alter) > 0:
                m21_root.getHigherEnharmonic(inPlace=True)
            return m21_root.name.replace('-', 'b')
        else:
            raise ValueError('Accidental must be sharp or flat.')

    def __repr__(self):
        return self.export_string()


def _compare_degree(a, b):
    if a[0] == '*':
        return _compare_degree(a[1:], b)

    if b[0] == '*':
        return _compare_degree(a, b[1:])

    i = 0
    while (a[i] == '#' or a[i] == 'b'):
        i += 1
    a_mod = a[:i]
    a_num = a[i:]

    i = 0
    while (b[i] == '#' or b[i] == 'b'):
        i += 1
    b_mod = b[:i]
    b_num = b[i:]

    if int(a_num) < int(b_num):
        return -1
    elif int(a_num) > int(b_num):
        return 1
    else:
        if len(b_mod) == 0:
            if len(a_mod) == 0:
                return 0
            elif a_mod[-1] == '#':
                return 1
            else:
                return -1
        elif b_mod[-1] == '#':
            if len(a_mod) == 0:
                return -1
            elif a_mod[-1] == 'b':
                return -1
            else:
                return _compare_degree(a_mod[:-1] + '0', b_mod[:-1] + '0')
        else:
            if len(a_mod) == 0:
                return 1
            elif a_mod[-1] == 'b':
                return _compare_degree(a_mod[:-1] + '0', b_mod[:-1] + '0')
            else:
                return 1


def _degree_to_halfsteps(degree):
    init_halfsteps = [0, 2, 4, 5, 7, 9, 11, 0, 2, 4, 5, 7, 9]
    val = init_halfsteps[int(degree[-1]) - 1]
    while '#' in degree or 'b' in degree:
        if '#' in degree:
            val += 1
            degree = degree[1:]
        if 'b' in degree:
            val -= 1
            degree = degree[1:]

    return val


def _degree_to_m21_interval(degree):
    generic_degree = degree.replace('#', '')
    generic_degree = generic_degree.replace('b', '')
    if generic_degree in ['1', '4', '5', '8', '11', '12']:
        if ('#' not in degree) and ('b' not in degree):
            return m21.interval.Interval('p' + degree)
        else:
            degree = degree.replace('#', 'a')
            degree = degree.replace('b', 'd')
            return m21.interval.Interval(degree)
    else:
        if ('#' not in degree) and ('b' not in degree):
            return m21.interval.Interval('M' + degree)
        else:
            if ('b' not in degree):
                degree = degree.replace('#', 'a')
                return m21.interval.Interval(degree)
            else:
                if degree[1] != 'b':
                    degree = degree.replace('b', 'm')
                    return m21.interval.Interval(degree)
                else:
                    degree = degree[1:].replace('b', 'd')
                    return m21.interval.Interval(degree)


def _m21_interval_to_degree(m21_int):
    if '-' in m21_int.directedName:
        m21_int = m21_int.complement

    degree = m21_int.name
    if m21_int.diatonicType > 13:
        raise ValueError('no interval over diatonic Type of 13 accepted')

    if str(m21_int.diatonicType) in ['1', '4', '5', '8', '11', '12']:
        if ('d' not in degree) and ('A' not in degree):
            return degree[1:]
        else:
            degree = degree.replace('A', '#')
            degree = degree.replace('d', 'b')
            return degree
    else:
        if 'M' in degree:
            return degree[1:]
        elif 'm' in degree:
            return 'b' + degree[1:]
        elif 'd' in degree:
            return 'b' * (degree.count('d') + 1) + degree.replace('d', '')
        else:
            return degree.replace('A', '#')
