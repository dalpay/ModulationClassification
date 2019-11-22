import numpy as np
from features import moments, cumulants, phase_stats, amplitude_stats, \
    higher_stats, iq_ratio, peak_to_mean, peak_to_rms


def load_iq_data(cfile_name, num_samps=-1):

    raw_nums = np.fromfile(cfile_name, dtype=np.float32, count=num_samps)
    iq_real = raw_nums[::2]
    iq_imag = raw_nums[1::2]
    iq_data = np.array(iq_real, dtype=np.complex64)
    iq_data.imag = iq_imag

    return iq_data

def process_features(frame, feature_types):

    feats = np.array([])

    if ('cumulants' in feature_types):
        feats = np.hstack((feats, cumulants(frame)))
    if ('moments' in feature_types):
        feats = np.hstack((feats, moments(frame, 6)))
    if ('amplitude_stats' in feature_types):
        feats = np.hstack((feats, amplitude_stats(frame)))
    if ('phase_stats' in feature_types):
        feats = np.hstack((feats, phase_stats(frame)))
    if ('higher_stats' in feature_types):
        feats = np.hstack((feats, higher_stats(frame)))
    if ('iq_ratio' in feature_types):
        feats = np.hstack((feats, iq_ratio(frame)))
    if ('peak_to_mean' in feature_types):
        feats = np.hstack((feats, peak_to_mean(frame)))
    if ('peak_to_rms' in feature_types):
        feats = np.hstack((feats, peak_to_rms(frame)))
    if (feats is None):
        print('Invalid feature types')
        return None

    return feats

def extract_features(files, mod_types, feature_types, snr_list=None,
    num_frames=100, frame_len=512, frame_step=256, verbose=False):

    num_features = 0
    if ('cumulants' in feature_types):
        num_features += 18 
    if ('moments' in feature_types):
        num_features += 12
    if ('amplitude_stats' in feature_types):
        num_features += 2
    if ('phase_stats' in feature_types):
        num_features += 2
    if ('higher_stats' in feature_types):
        num_features += 4
    if ('iq_ratio' in feature_types):
        num_features += 1
    if ('peak_to_mean' in feature_types):
        num_features += 1
    if ('peak_to_rms' in feature_types):
        num_features +=1 

    iq_data = dict()
    for mod_type, files in files.items():

        iq_data[mod_type] = np.array([])
        num_samps = 2*frame_len*num_frames
        
        file = iter(files)
        while (len(iq_data[mod_type]) < num_samps):
            
            num_samps_left = num_samps - len(iq_data[mod_type])
            
            try:
                iq_data[mod_type] = np.hstack((iq_data[mod_type], 
                    load_iq_data(next(file), num_samps=num_samps_left)))
            except StopIteration:
                if (verbose):
                    print('Not enough data available for {0} class'.format(mod_type))
                break

    features = dict.fromkeys(iq_data.keys(), np.empty((0, num_features)))
    
    percent_increment = 10

    for mod_type, signal in iq_data.items():
        
        next_percent = percent_increment
        ind = 0
        
        while (ind + frame_len < len(signal)):

            frame = np.array(signal[ind : ind + frame_len], dtype=np.complex128)
            
            if (snr_list is not None):
                for snr in snr_list:
                    
                    signal_power = np.mean(np.abs(frame)**2)
                    noise_power = signal_power / 10**(snr / 10)
                    new_frame = np.array(frame)
                    new_frame.real += np.random.normal(0, noise_power, frame.shape)
                    new_frame.imag += np.random.normal(0, noise_power, frame.shape)

                    feats = process_features(new_frame, feature_types)
                    features[mod_type] = np.vstack((features[mod_type], feats))        

            else:
                feats = process_features(frame, feature_types)
                features[mod_type] = np.vstack((features[mod_type], feats))
            
            percent_processed = (ind + frame_len) / len(signal) * 100
            if (verbose and percent_processed > next_percent):
                print('Extracted {0:.0f} percent of {1} class'.format(percent_processed, mod_type.upper()))
                next_percent += percent_increment

            ind += frame_step
    
    return features

def convert_features(features_dict):

    mod_types = list(features_dict.keys())
    features = np.empty((0, features_dict[mod_types[0]].shape[1]))
    labels = np.array([], dtype=np.int64)

    for mod_type, feats in features_dict.items():

        features = np.vstack((features, feats))
        new_labels = mod_types.index(mod_type) * np.ones(feats.shape[0], dtype=np.int32)
        labels = np.hstack((labels, new_labels))

    return (features, labels)
