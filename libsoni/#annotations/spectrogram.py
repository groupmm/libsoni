import numpy as np


def sonify_spectrogram(chroma_data, N, frame_rate, Fs, fading_msec=5):
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
