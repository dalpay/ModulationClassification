import numpy as np
import matplotlib.pyplot as plt
from extract_features import extract_features, convert_features
from FeedforwardNetwork import FeedforwardNetwork


def main():

    data_path = '../noiseless_data/'
    model_path = '../model_5/model_5'
    mod_types = ['gfsk', 'gmsk', 'qam4', 'qam16', 'qam64', 'psk2', 'psk4', 'psk8']
    feature_types = ['cumulants', 'amplitude_stats', 'phase_stats']

    num_frames = int(1e3)
    frame_len = 2048
    frame_step = 256
    snr_list = np.arange(-5, 11, 1)

    print('Loading model')
    fnn = FeedforwardNetwork(model_path, 3)

    print('Preparing files')
    files = dict()
    for mod_type in mod_types:

        files[mod_type] = list()    
        for k in range(1, 6):

            files[mod_type].append(data_path + mod_type + '_' + str(k) + '.txt')

    accuracies = list()
    for snr in snr_list:
        
        print('SNR = {0} dB'.format(snr))

        print('Extracting features')
        features_dict = extract_features(files, mod_types, feature_types=feature_types, 
            frame_len=frame_len, frame_step=frame_step, num_frames=num_frames, 
            snr_list=[snr], verbose=False)
        print('Number of features: {0}'.format(features_dict[mod_types[0]].shape[1]))

        print('Converting features')
        (features, labels) = convert_features(features_dict)
        print('Number of samples: {0}'.format(features.shape[0]))
        
        print('Testing model')
        num_correct = 0
        for feature, label in zip(features, labels):
            
            prediction = fnn.predict(feature)
            if (prediction == label):
                num_correct += 1

        accuracy = num_correct / len(labels)
        accuracies.append(accuracy)
        print('Accuracy: {:.2f}'.format(accuracy))

    plt.figure()
    plt.bar(snr_list, accuracies)
    plt.xticks(snr_list)
    plt.xlabel('SNR (dB)') 
    plt.ylabel('Accuracy') 
    plt.title('')
    plt.show()

if __name__ == '__main__':

    main()

