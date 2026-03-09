import os
import pickle
import numpy  as np


"""

"""



def check_data_quality(left_data, right_data, base_name):
    '''
    Detect and count the number of saturation points in the data
     of the patient's two feet.
    '''
    print (f"\nData quality check for {base_name}:")
    print(f"  Number of data points: {len(left_data)}")
    print(f"  Minimum value: {left_data.min()}")
    print(f"  Maximum value: {left_data.max()}")
    print(f"  Mean: {left_data.mean():.2f}")
    print(f"  Standard deviation: {left_data.std():.2f}")

    """
    -32768 is the minimum value of a 16-bit signed integer, and 32767 is the maximum value. This usually indicates:
    1. The sensor has not been calibrated correctly.
    2. There might be a problem with the data quality of the right foot.
    3. Or it could be a special zero-point setting. 

    *** However, even if it's not -32768 and 32767, values like -16030 or -686 would still cause significant disturbances to the result.
    **** Compared to *.ts data, it is much more orderly. 
    """
    # Check separately whether there are saturation values on the left and right feet
    n_saturation_l = np.sum(left_data==32767) + np.sum(left_data== -32768)
    n_saturation_r = np.sum(right_data==32767) + np.sum(right_data== -32768)

    n_saturation_lr =  n_saturation_l + n_saturation_r

    if n_saturation_lr>0:
        print (f'Warning: {n_saturation_lr} saturation values have been detected.')

    return n_saturation_lr


def check_data_baseCalibValue(left_base, right_base, left_calib, right_calib, base_name):
    """
    Check and fix abnormal calibration parameters
    """
    print (f"\nQuality check of calibration parameters for {base_name}:")

    l_saturation_base_value = (left_base==-32768) + (left_base==32767)
    r_saturation_base_value =  (right_base==-32768) + (right_base==32767)
    lr_saturation_base_value =  l_saturation_base_value + r_saturation_base_value

    if lr_saturation_base_value>0:
        print (f"\nThe base value of {base_name} has reached its saturation point!")

    l_saturation_calib_value = (left_calib==-32768) + (left_calib==32767)
    r_saturation_calib_value =  (right_calib==-32768) + (right_calib==32767)
    lr_saturation_calib_value =  l_saturation_calib_value + r_saturation_calib_value

    if lr_saturation_calib_value>0:
        print (f"\nThe calibration value of {base_name} has reached its saturation limit!")


    return lr_saturation_base_value, lr_saturation_calib_value


#

def tj_lr_data_quality(pkl_path):

    total_lr_n_saturation = 0
    total_lr_saturation_base_value = 0
    total_lr_saturation_calib_value = 0

    for pkl_file in os.listdir(pkl_path):
        # print ('pkl_file:', pkl_file)
        if 'als12.pkl'==pkl_file:
            continue
        with open (os.path.join(pkl_path, pkl_file),'rb') as f:
            lr_data_dic = pickle.load(f)


        base_name =  lr_data_dic['subject']
        left_data =  lr_data_dic['left_data']
        right_data = lr_data_dic['right_data']
        # base value and calibration for left and right
        left_base = lr_data_dic['left_base']
        right_base = lr_data_dic['right_base']
        left_calib = lr_data_dic['left_calib']
        right_calib = lr_data_dic['right_calib']

        # Simple detection of data quality for both left and right feet
        lr_n_saturation = check_data_quality(left_data, right_data, base_name)

        # Calculate the cumulative value
        total_lr_n_saturation = total_lr_n_saturation + lr_n_saturation


        # Quality inspection of the base values and calibration values of the left and 
          ## right feet
        lr_saturation_base_value, lr_saturation_calib_value = check_data_baseCalibValue(left_base, right_base, left_calib, right_calib,  base_name)

        # Calculate cumulative value
        total_lr_saturation_base_value = total_lr_saturation_base_value + lr_saturation_base_value
        total_lr_saturation_calib_value = total_lr_saturation_calib_value + lr_saturation_calib_value


    lr_saturation_dic = {'ch_ac_results4lr': total_lr_n_saturation}

    bc_saturation_dic = {
    'total_lr_saturation_base_value': total_lr_saturation_base_value,
    'total_lr_saturation_calib_value': total_lr_saturation_calib_value
    }

    return lr_saturation_dic, bc_saturation_dic





'''
the qualities of park + als + hunt + control
'''
if __name__ == "__main__":
    pkl_path = "D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\gait64_neodegen_pkls"
    lr_saturation_dic,  bc_saturation_dic = tj_lr_data_quality(pkl_path)

    for key, value in lr_saturation_dic.items():
        print ('xx_64_13:', key)
        print ('The cumulative number of left and right foot jump variations in the xx_64_13 dataset is:', value)


    for key_, value in bc_saturation_dic.items():
        print  ('xx_64_13:', key_)
        print  ('The cumulative number of jumps in the base values and corrected values for both the left and right feet in the xx_64_13 dataset is:', value)
