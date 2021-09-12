import os
import librosa
import json
import math

DATASET_PATH = 'test_data'
JSON_PATH = 'data.json'

SAMPLE_RATE = 22050

def add_noise(data):
        noise = np.random.randn(len(data))
        data_noise = data + (0.005 * noise)
        print(data_noise)
        return data_noise

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, noise=False):
    
    # dictionary to store data
    data = {
        'mapping': [],
        'mfcc': [], # training input
        'labels': [], # expected values
    }
    
    samples_per_file = SAMPLE_RATE * 1#duration
    expected_num_mfcc_vectors = math.ceil(samples_per_file / hop_length)
    
    # Each 1 second sample will preprocess into a 13x44 matrix
    #loop through all the categories    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # want to ensure we're not at the root level
        if dirpath is not dataset_path:
            
            # save the semantic label
            dirpath_components = dirpath.split('/') # TESS/OAF_angry
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            
            # process files for each emotion.
            for f in filenames:
                
                # load the audio file
                file_path =  os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                signal = signal[-sr:]
                
                if noise:
                    signal = add_noise(signal)
                
                # extract mfcc from each file and store data
                mfcc = librosa.feature.mfcc(signal, sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)
                mfcc =  mfcc.T
                
                data['mfcc'].append(mfcc.tolist())
                data['labels'].append(i-1)
                print("{}".format(file_path))

    fp = open(json_path, 'w')
    json.dump(data, fp, indent=4)
    fp.close()

def main():
    save_mfcc(DATASET_PATH, JSON_PATH, noise=True)

main()