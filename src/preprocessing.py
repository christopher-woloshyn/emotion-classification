import json
import os
import sys

import librosa
import numpy as np

from visualizations import visualize_all

SAMPLE_RATE = 22050

    
def add_noise(data, amount=0.001):
    """Add random uniform noise to the existing waveform."""
    data += amount * np.random.randn(len(data))

    return data

def save_mfcc(
    dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, noise=0.
    ):
    """Preprocesses sound data using MFCC methods. Stores results in JSON."""

    # Create the output dictionary.
    data = {
        'mapping': [],
        'mfcc': [], # training input
        'labels': [], # expected values
    }

    # Each 1 second sample will preprocess into a 13x44 matrix.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # Want to ensure we're not at the root level.        
        if dirpath is not dataset_path:
            
            # Save the semantic label.
            dirpath_components = dirpath.split('/') # TESS/OAF_angry
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            print(f'Processing: {semantic_label}\n')
            
            # Process files for each emotion.
            for f in filenames:
                # Load the audio file.
                file_path =  os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                signal = signal[-sr:]
                
                # Pass noise over the audio file.
                if noise:
                    signal = add_noise(signal, amount=noise)
                
                # Extract MFCC from each file and store the result.
                mfcc = librosa.feature.mfcc(
                    signal,
                    sr=sr,
                    n_fft=n_fft,
                    n_mfcc=n_mfcc,
                    hop_length=hop_length,
                    )
                mfcc =  mfcc.T
                
                data['mfcc'].append(mfcc.tolist())
                data['labels'].append(i-1)

    fp = open(json_path, 'w')
    json.dump(data, fp, indent=2)
    fp.close()

def main():
    noises = {"no_noise": 0, "light_noise": 0.001, "heavy_noise": 0.01}
    dataset_path = f'{sys.path[0]}/../data/TESS'

    for noise in noises.keys():
        print(f'Processing: {noise}')
        json_path = f'{sys.path[0]}/../data/json/{noise}.json'
        save_mfcc(dataset_path, json_path, noise=noises[noise])

        # Visualize the same file for all noise types!
        # visualize_all()

if __name__ == '__main__':
    main()
