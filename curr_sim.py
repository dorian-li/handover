import numpy as np


def generate_modified_square_wave(
    sampling_rate=1000,
    high_duration=0.5,
    low_duration=0.5,
    amplitude=1.0,
    duration=10,
    noise_level=0.1,
):
    """
    Generate a modified square wave signal where the high duration and low duration are independent.

    Parameters:
    - sampling_rate: int, samples per second.
    - high_duration: float, duration of the high state in seconds.
    - low_duration: float, duration of the low state in seconds.
    - amplitude: float, amplitude of the square wave.
    - duration: float, total duration of the signal in seconds.
    - noise_level: float, standard deviation of Gaussian noise added to the signal.

    Returns:
    - t: numpy array, time vector.
    - signal: numpy array, generated square wave signal with noise.
    """
    # Time vector from 0 to duration with specified sampling rate
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generate the modified square wave signal
    cycle_duration = high_duration + low_duration
    cycles = t // cycle_duration
    signal = amplitude * ((t - cycles * cycle_duration) < high_duration).astype(int)

    signal = signal[850:]
    signal = np.append(signal, np.zeros(850))

    # Add Gaussian noise
    noise = noise_level * np.random.normal(size=t.shape)
    signal_with_noise = signal + noise * 1.1

    return t, signal_with_noise
