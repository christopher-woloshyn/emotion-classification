import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


def add_noise(data, noise):
    """Add random uniform noise to the existing waveform."""
    data += noise * np.random.randn(len(data))

    return data
    
def see_waveform(file, noise):
    """Plots raw waveform data with respect to time."""
    signal, sr = librosa.load(file)
    signal = signal[-sr:]
    if noise:
        signal = add_noise(signal, noise)
    
    name = file.split('/')[-1]
    
    librosa.display.waveplot(signal)
    plt.title(name + ': Raw Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def see_fourier_transform(file, noise, log=False, sr=22050):
    """Plots FFT transformed data."""
    signal, sr = librosa.load(file)
    signal = signal[-sr:]
    if noise:
        signal = add_noise(signal, noise)
        
    name = file.split('/')[-1]
    
    fft = np.fft.fft(signal)
    
    mag = np.abs(fft)
    freq = np.linspace(0, sr, len(mag))
    
    left_freq = freq[:int(len(freq) / 2)]
    left_mag = mag[:int(len(mag) / 2)]

    if log:
        plt.semilogx(left_freq, left_mag)
        plt.title(name + ': Fast Fourier Transform (semi-log scale)')
        
    else:
        plt.plot(left_freq, left_mag)
        plt.title(name + ': Fast Fourier Transform')

    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

def see_spectrogram(file, noise, log=True, sr=22050, n_fft=2048, hop_length=512):
    """Plots spectrogram of the waveform data after the FFT."""
    signal, sr = librosa.load(file)
    signal = signal[-sr:]
    if noise:
        signal = add_noise(signal, noise)
        
    name = file.split('/')[-1]
    
    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(stft)
    
    if log:
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        
        librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
        plt.title(name + ': Spectrogram on DB scale')
        
    else:
        librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
        plt.title(name + ': Spectrogram on linear scale')
        
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    plt.show()

def see_mfcc(file, noise, sr=22050, n_fft=2048, hop_length=512):
    """Plots MFCC matrix after all components of the transform have occured."""
    signal, sr = librosa.load(file)
    signal = signal[-sr:]
    if noise:
        signal = add_noise(signal, noise)
        
    name = file.split('/')[-1]
    
    features = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length)

    librosa.display.specshow(features, sr=sr, hop_length=hop_length)
    plt.title(name + ': MFCC matrix')
    plt.xlabel('Time')
    plt.ylabel('MFCC')
    plt.colorbar()
    plt.show()
    
def visualize_all(file, noise=0):
    """Plots waveform between all stages of the preprocessing."""
    plt.style.use('ggplot')
    see_waveform(file, noise)
    see_fourier_transform(file, noise)
    see_fourier_transform(file, noise, log=True)
    see_spectrogram(file, noise, log=False)
    see_spectrogram(file, noise)
    see_mfcc(file, noise)
