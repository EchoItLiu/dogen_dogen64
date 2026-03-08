import os
import numpy as np
import pandas as pd
import pickle
import sys
import json




def parse_gait_binary(file_path, base, calibrationFactor, sampleRate=300):
    """
    Parse the binary file of gait data 

    Parameters:
    file_path: Binary file path
    base_line: Baseline offset (read from .hea file)
    calibrationFactor: Calibration factor (read from .hea file)
    sampleRate: Sampling frequency (Hz) 

    """
    # Binary File Reading
    with open(file_path, 'rb') as f:

        # ADC value = physical value × gain + offset #

        data = f.read()
    
        values = np.frombuffer(data, dtype='<i2')



        # Physical value = (ADC value [values] - offset [base]) / gain [calibration ≈ gain]
        #
        #
        calibrated = (values - base) / calibrationFactor

        # Generate timeline
        time_axis = np.arange(len(calibrated)) / sampleRate

        return time_axis, calibrated



def normalize_signals(left_data, right_data, method='minmax'):

    if method == 'minmax':
        left_norm = (left_data - left_data.min()) / (left_data.max()-left_data.min())
        right_norm = (right_data - right_data.min()) / (right_data.max() - right_data.min())


    elif  method == 'zscore':
        left_norm = (left_data - left_data.mean()) /  left_data.std()
        right_norm = (right_data - right_data.min()) / right_data.std()

    elif method == 'max':
        left_norm = left_data / left_data.max()
        right_norm = right_data / right_data.max()

    elif method == 'relative':
        left_norm = left_data - left_data.mean()
        right_norm = right_data - right_data.mean()


    elif method == 'peak':
        left_norm = left_data / left_data.max()
        right_norm = right_data / right_data.max()

    return left_norm, right_norm



def prepend_gait64_dataset(data_dir):
    records = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.hea'):
            base_name = filename.replace('.hea','')
            hea_path = os.path.join(data_dir, filename)

            with open(hea_path,'r') as f:
                lines = f.readlines()


            for line in lines:
                if '.let' in line:
                    parts = line.split()
                    left_base = int(parts[5])
                    left_calib = int(parts[6])
                    sample_rate = int(parts[2])

                elif '.rit' in line:
                    parts = line.split()
                    right_base = int(parts[5])
                    right_calib = int(parts[6])

        let_path = os.path.join(data_dir, f"{base_name}.let")
        rit_path = os.path.join(data_dir, f"{base_name}.rit")


        if os.path.exists(let_path) and os.path.exists(rit_path):
            time_left, left_data = parse_gait_binary(let_path, left_base, left_calib, sample_rate)
            time_right, right_data = parse_gait_binary(rit_path, right_base, right_calib, sample_rate)


        left_data, right_data = normalize_signals(left_data, right_data, method='peak')

        print(f"Left foot data point count: {len(left_data)}")
        print(f"Data point number of the right foot: {len(right_data)}")
        print(f"Sampling time: {time_left[-1]:.2f} sec")
        print(f"Left foot pressure range: {left_data.min():.4f} ~ {left_data.max():.4f}")
        print(f"Range of pressure on the right foot: {right_data.min():.4f} ~ {right_data.max():.4f}")

        #

        ts13_list = []
        print ('filename_sets:', filename)
        print ('---1---')

        ts_path = os.path.join(data_dir, f"{base_name}.ts")
        print ('ts_path', ts_path)

        with open(ts_path,'r') as f:
            lines = f.readlines()

        for line in lines:
            ts13_list.append(line.split())

        ts13_array = np.array(ts13_list)
        #
        print ('ts13_array_shape:', ts13_array.shape)

        # 12
        patient_dic = {
            'subject': base_name,
            'time_left': time_left,
            'time_right': time_right,
            'left_data': left_data,
            'right_data': right_data,
            'sample_rate': sample_rate,
            'duration': time_left[-1],
            'n_samples': len(left_data),
            'ts13_array': ts13_array,
            'left_base': left_base,
            'right_base': right_base,
            'left_calib': left_calib,
            'right_calib': right_calib
        }

        patient_dic_ls =  {
            'subject': base_name,
            'time_left': list(time_left),
            'time_right': list(time_right),
            'left_data': list(left_data),
            'right_data': list(right_data),
            'sample_rate': sample_rate,
            'duration': time_left[-1],
            'n_samples': len(left_data),
            # 'ts13_array': [ts for ts in ts13_array],
            'left_base': left_base,
            'right_base': right_base,
            'left_calib': left_calib,
            'right_calib': right_calib
        }

        with open(os.path.join(data_dir, 'gait64_neodegen_pkls',f"{base_name}.pkl"), 'wb') as f:
        # with open(os.path.join(r'D:\gait-in-neurodegenerative-disease-database-1.0.0\backup_pkls',f"{base_name}.pkl"), 'wb') as f:

            pickle.dump(patient_dic, f)



        with open(os.path.join(data_dir,'log_board', 'gait-64_lan_log.json'),'w', encoding='utf-8') as f:
            json.dump(patient_dic_ls, f, ensure_ascii = False, indent=4)

    return 0

if __name__=="__main__":
    gait64_degen_path = "D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0"

    prepend_gait64_dataset(gait64_degen_path)
