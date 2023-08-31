import numpy as np
from scipy.signal import sawtooth


def sonify_chroma(chromagram):
    # TODO: implement (;
    return



def shepard_tone(frequencies, amplitude, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tones = np.zeros_like(t)
    for freq in frequencies:
        tones += amplitude * sawtooth(2 * np.pi * freq * t)
    return tones


def sonify_chroma_matrix(chroma_matrix, sample_rate=22050, base_frequency=220, duration_per_frame=0.5):
    num_frames, num_notes = chroma_matrix.shape
    chroma_notes = len(chroma_matrix[0])

    output = np.zeros(0)

    for frame_idx in range(num_frames):
        active_notes = [note_idx for note_idx, amplitude in enumerate(chroma_matrix[frame_idx]) if amplitude > 0]
        frequencies = [base_frequency * 2 ** (note_idx / 12.0) for note_idx in active_notes]
        frame_wave = shepard_tone(frequencies, 0.5, duration_per_frame, sample_rate)
        output = np.concatenate((output, frame_wave))

    return output


def _list_to_chromagram(note_list, num_frames, frame_rate):
    """Create a chromagram matrix from a list of note events

    Parameters
    ----------
    note_list : List
        A list of note events (e.g. gathered from a CSV file by libfmp.c1.pianoroll.csv_to_list())

    num_frames : int
        Desired number of frames for the matrix

    frame_rate : float
        Frame rate for C (in Hz)

    Returns
    -------
    C : NumPy Array
        Chromagram matrix
    """
    C = np.zeros((12, num_frames))
    for l in note_list:
        start_frame = max(0, int(l[0] * frame_rate))
        end_frame = min(num_frames, int((l[0] + l[1]) * frame_rate) + 1)
        C[int(l[2] % 12), start_frame:end_frame] = 1
    return C


def _generate_shepard_tone(chromaNum, Fs, N, weight=1, Fc=440, sigma=15, phase=0):
    """
    inputs:
        chromaNum: 1=C,...
        Fs: sampling frequency
        N: desired length (in samples)
        weight: scaling factor [0:1]
        Fc: frequency for A4
        sigma: parameter for envelope of Shepard tone
        fading: fading at the beginning and end of the tone (in ms)
    output:
        shepard tone
    """
    tone = np.zeros(N)
    # Envelope function for Shepard tones
    p = 24 + chromaNum
    if p > 32:
        p = p - 12
    while p < 108:
        scale_factor = 1 / (np.sqrt(2 * np.pi) * sigma)
        A = scale_factor * np.exp(-(p - 60) ** 2 / (2 * sigma ** 2))
        f_axis = np.arange(N) / Fs
        sine = np.sin(2 * np.pi * np.power(2, ((p - 69) / 12)) * Fc * (f_axis + phase))
        tmp = weight * A * sine
        tone = tone + tmp
        p = p + 12
    return tone


def sonify_chromagram(chroma_data, N, frame_rate, Fs, fading_msec=5):
    """Sonify the chroma features from a chromagram

    Parameters
    ----------
    chroma_data : NumPy Array
        A chromagram (e.g. gathered from a list of note events by list_to_chromagram())

    N : int
        Length of the sonification (in samples)

    frame_rate : float
        Frame rate for P (in Hz)

    Fs : float
        Sampling frequency (in Hz)

    fading_msec : float
        The length of the fade in and fade out for sonified tones (in msec)

    Returns
    -------
    chroma_son : NumPy Array
        Sonification of the chromagram
    """

    # empty chromagram
    chroma_son = np.zeros((N,))

    fade_sample = int(fading_msec / 1000 * Fs)

    # iterate chromagram rows

    for i in range(12):

        # Case: sum of entries in i-th row of chromagram is greater zero
        if np.sum(np.abs(chroma_data[i, :])) > 0:

            # generate a shepard tone with length of sonification (chromagram)

            shepard_tone = _generate_shepard_tone(i, Fs, N)

            # empty weights
            weights = np.zeros((N,))

            # iterate chroma columns
            for j in range(chroma_data.shape[1]):

                # Case: entry [i,j] is greater zero
                if np.abs(chroma_data[i, j]) > 0:
                    start = min(N, max(0, int((j - 0.5) * Fs / frame_rate)))
                    end = min(N, int((j + 0.5) * Fs / frame_rate))
                    fade_start = min(N, max(0, start + fade_sample))
                    fade_end = min(N, end + fade_sample)

                    weights[fade_start:end] += chroma_data[i, j]
                    weights[start:fade_start] += np.linspace(0, chroma_data[i, j], fade_start - start)
                    weights[end:fade_end] += np.linspace(chroma_data[i, j], 0, fade_end - end)

            chroma_son += shepard_tone * weights

    chroma_son = chroma_son / np.max(np.abs(chroma_son))

    return chroma_son


def sonify_chromagram_with_noc(chroma_data,
                               H,
                               # frame_rate,
                               fs=44100,
                               f_tuning=440.0,
                               fading_msec=5):
    """Sonify the chroma features from a chromagram

    Parameters
    ----------
    chroma_data : NumPy Array
        A chromagram (e.g. gathered from a list of note events by list_to_chromagram())

    N : int
        Length of the sonification (in samples)

    frame_rate : float
        Frame rate for P (in Hz)

    Fs : float
        Sampling frequency (in Hz)

    fading_msec : float
        The length of the fade in and fade out for sonified tones (in msec)

    Returns
    -------
    chroma_son : NumPy Array
        Sonification of the chromagram
    """

    chroma_sonification = np.zeros((chroma_data.shape[1] * H))

    # iterate pitch classes

    for pitch_class in range(12):

        shepard_frequencies = f_tuning * 2 ** ((np.arange(start=24 + pitch_class, stop=127, step=12) - 69) / 12)

        weights = chroma_data[pitch_class, :]
        weights_stretched = np.zeros_like(chroma_sonification)

        for m in range(chroma_data.shape[1] - 1):
            weights_stretched[m * H:m + 1 * H] = weights[m]

        for shepard_frequency in shepard_frequencies:

            phase = 0
            phase_result = []

            for j, weight in enumerate(weights_stretched):
                phase_step = 2 * np.pi * shepard_frequency * 1 / fs
                phase += phase_step
                phase_result.append(phase)

        chroma_sonification += np.sin(phase_result)

    return chroma_sonification
    #     for i, shepard_frequency in enumerate(shepard_frequencies):
    #         phase = 0
    #         phase_result = []
    #         for frequency in frequency_vector:
    #             phase_step = 2 * np.pi * frequency * harmonic * 1 / fs
    #             phase += phase_step
    #             phase_result.append(phase)
    #         # generate a shepard tone with length of sonification (chromagram)
    #         y += np.sin(phase_result) #* harmonics_amplitudes[i]
    #
    #         N = int(dur * Fs)
    #         t = np.arange(N) / Fs
    #         freqs = f_tuning * 2 ** ((pitch_class + np.arange(12) * 12 - 69) / 12)
    #         f_log = 2 * np.logspace(1, 4, 20000)
    #         f_lin = np.linspace(20, 20000, 20000)
    #         f_center_lin = np.argmin(np.abs(f_log - f_center))
    #         weights = np.exp(- (f_lin - f_center_lin) ** 2 / (1.4427 * ((octave_cutoff * 2) * 1000) ** 2))
    #         y = np.zeros(N)
    #
    #         for freq in freqs:
    #             y += weights[np.argmin(np.abs(f_log - freq))] * np.sin(2 * np.pi * freq * t)
    #
    #         fade_samples = int(fade_dur * Fs)
    #         y[0:fade_samples] *= np.linspace(0, 1, fade_samples)
    #         y[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    #         y = amp * (y / np.max(y))
    #         shepard_tone = _generate_shepard_tone(i, Fs, N)
    #
    #         # empty weights
    #         weights = np.zeros((N,))
    #         tone = np.zeros(N)
    #         # Envelope function for Shepard tones
    #         p = 24 + chromaNum
    #         if p > 32:
    #             p = p - 12
    #         while p < 108:
    #             scale_factor = 1 / (np.sqrt(2 * np.pi) * sigma)
    #             A = scale_factor * np.exp(-(p - 60) ** 2 / (2 * sigma ** 2))
    #             f_axis = np.arange(N) / Fs
    #             sine = np.sin(2 * np.pi * np.power(2, ((p - 69) / 12)) * Fc * (f_axis + phase))
    #             tmp = weight * A * sine
    #             tone = tone + tmp
    #             p = p + 12
    #
    #         # iterate chroma columns
    #         for j in range(chroma_data.shape[1]):
    #
    #             # Case: entry [i,j] is greater zero
    #             if np.abs(chroma_data[i, j]) > 0:
    #                 start = min(N, max(0, int((j - 0.5) * Fs / frame_rate)))
    #                 end = min(N, int((j + 0.5) * Fs / frame_rate))
    #                 fade_start = min(N, max(0, start + fade_sample))
    #                 fade_end = min(N, end + fade_sample)
    #
    #                 weights[fade_start:end] += chroma_data[i, j]
    #                 weights[start:fade_start] += np.linspace(0, chroma_data[i, j], fade_start - start)
    #                 weights[end:fade_end] += np.linspace(chroma_data[i, j], 0, fade_end - end)
    #
    #         chroma_son += shepard_tone * weights
    #
    # chroma_son = chroma_son / np.max(np.abs(chroma_son))
