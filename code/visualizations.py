import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


def add_noise(data, amount):
        data += amount * np.random.randn(len(data))
        print(data)
        return data
    
def see_waveform(file, noise, amount):
    signal, sr = librosa.load(file)
    signal = signal[-sr:]
    if noise:
        signal = add_noise(signal, amount)
    
    name = file.split('/')[-1]
    
    librosa.display.waveplot(signal)
    plt.title(name + ': Raw Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def see_fourier_transform(file, noise, amount, log=False, sr=22050):
    signal, sr = librosa.load(file)
    signal = signal[-sr:]
    if noise:
        signal = add_noise(signal, amount)
        
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

def see_spectrogram(file, noise, amount, log=True, sr=22050, n_fft=2048, hop_length=512):
    signal, sr = librosa.load(file)
    signal = signal[-sr:]
    if noise:
        signal = add_noise(signal, amount)
        
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

def see_mfcc(file, noise, amount, sr=22050, n_fft=2048, hop_length=512):
    signal, sr = librosa.load(file)
    signal = signal[-sr:]
    if noise:
        signal = add_noise(signal, amount)
        
    name = file.split('/')[-1]
    
    features = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length)

    librosa.display.specshow(features, sr=sr, hop_length=hop_length)
    plt.title(name + ': MFCC matrix')
    plt.xlabel('Time')
    plt.ylabel('MFCC')
    plt.colorbar()
    plt.show()
    
def visualize_all(file, noise=True, amount=0.001):
    plt.style.use('ggplot')
    see_waveform(file, noise, amount)
    see_fourier_transform(file, noise, amount)
    see_fourier_transform(file, noise, amount, log=True)
    see_spectrogram(file, noise, amount, log=False)
    see_spectrogram(file, noise, amount)
    see_mfcc(file, noise, amount)
