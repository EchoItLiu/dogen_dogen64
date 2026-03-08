
from scipy import stats
from scipy.signal import find_peaks
import pickle
import numpy as np
import os



def load_pkl_gait64_one_case(pkl_path):

    for pkl_file_name in os.listdir(pkl_path):
        if 'park15' in pkl_file_name:
            with open (os.path.join(pkl_path, pkl_file_name), 'rb') as f:
                dic_park15 = pickle.load(f)

            left_signal_park15 = dic_park15['left_data']
            right_signal_park15 = dic_park15['right_data']
            sample_rate_park15 = dic_park15['sample_rate']

    return left_signal_park15, right_signal_park15, sample_rate_park15



def extract_gait_features(left_signal_park15, right_signal_park15, sample_rate = 300):

    features = {}
    # 1. Basic Statistical Characteristics
    features['left_mean'] = left_signal_park15.mean()
    features['left_std'] = left_signal_park15.std()
    features['left_max'] = left_signal_park15.max()
    features['left_min'] = left_signal_park15.min()
    #
    features['right_mean'] = right_signal_park15.mean()
    features['right_std'] = right_signal_park15.std()
    features['right_max'] = right_signal_park15.max()
    features['right_min'] = right_signal_park15.min()

    # 2. Gait Symmetry Characteristics
    features['asymmetry_index'] = np.abs(left_signal_park15.mean() - right_signal_park15.mean()) / (left_signal_park15.mean() + right_signal_park15.mean())


    features['corrlelation'] = np.corrcoef(left_signal_park15, right_signal_park15)[0,1]

    # Frequency domain feature extraction
    left_fft = np.fft.fft(left_signal_park15)
    right_fft = np.fft.fft(right_signal_park15)
    freqs = np.fft.fftfreq(len(left_signal_park15), 1/sample_rate)




    # 5、Clock frequency (step frequency)
    left_power = np.abs(left_fft[:len(freqs)//2])
    right_power = np.abs(right_fft[:len(freqs)//2])
    features['left_dominant_freq'] = freqs[:len(freqs)//2][np.argmax(left_power)]
    features['right_dominant_freq'] = freqs[:len(freqs)//2][np.argmax(right_power)]

    # 6、 Gait cycle detection (based on peak values)
    ##
    left_peaks, _ = find_peaks(left_signal_park15, height = left_signal_park15.max() * 0.5)
    right_peaks, _ = find_peaks(right_signal_park15, height = right_signal_park15.max()*0.5)

    if len(left_peaks) > 1:
        '''
        2	Left Stride Interval (sec)
        3	Right Stride Interval (sec)
        '''
        # Calculate the difference
        left_stride_intervals = np.diff(left_peaks) / sample_rate
        features['left_stride_mean'] =left_stride_intervals.mean()
        features['left_stride_cv'] = left_stride_intervals.std() / left_stride_intervals.mean()

    if len(right_peaks) > 1:
        right_stride_intervals = np.diff(right_peaks) / sample_rate
        features['right_stride_mean'] = right_stride_intervals.mean()
        features['right_stride_cv'] = right_stride_intervals.std() / right_stride_intervals.mean()

    return features




if __name__== "__main__":
    pkl_path = "D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\gait64_neodegen_pkls"

    left_signal_park15, right_signal_park15, sample_rate_park15 = load_pkl_gait64_one_case(pkl_path)

    features = extract_gait_features(left_signal_park15, right_signal_park15, sample_rate_park15)

    print ("Extracted gait features:")

    for key, value in features.items():
        if isinstance(value, float):
            print (f"{key}: {value:.4f}")
        else:
            print (f"{key}: {value}")
