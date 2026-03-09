
"""
i)...
    ii)...
        ***
"""


import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

def ndstrarr2ndarray(str_nd_arr):
    int_nd_arr  = np.zeros((str_nd_arr.shape[0], str_nd_arr.shape[1]))
    for i in range(str_nd_arr.shape[0]):
        for j in range(str_nd_arr.shape[1]):
            int_nd_arr[i][j] = float(str_nd_arr[i][j])
    return int_nd_arr



# als12.let/rit;  hunt12.let/rit
def load_pkl_gait64_two_cases(path_dir_pkl):

    for sc_patient_candi in os.listdir(path_dir_pkl):

        # Select patient with ALS of No. 12
        # print ('---3---')
        if 'als13' in sc_patient_candi:
            # print ('---4---')
            # print ('path_check:', os.path.join(path_dir_pkl, sc_patient_candi))
            with open(os.path.join(path_dir_pkl, sc_patient_candi), 'rb') as file:
                als12_dic = pickle.load(file)
            als12_time_left = als12_dic['time_left']
            als12_time_right = als12_dic['time_right']

            als12_left_foot = als12_dic['left_data']

            # print ('als12_left_foot', als12_left_foot)
            als12_right_foot = als12_dic['right_data']

            als12_ts13 = als12_dic['ts13_array']

            # print ('als12_ts13_I', als12_ts13)
            als12_ts13_float = ndstrarr2ndarray(als12_ts13)

            # print  ('als12_ts13_II', als12_ts13_float)

            ts_100_indice_ = [5,6,9,10,12]

            # print ('als12_ts13_type:', type(als12_ts13_float))
            # print ('als12_ts13[:, k]_gtype:', type(als12_ts13_float[:, 5]) )
            # print ('als12_ts13[:, k]_gshape:', als12_ts13_float[:, 5].shape )

            for k in ts_100_indice_:
                als12_ts13_float[:, k] = als12_ts13_float[:, k] / 100

            # print ('als12_ts13_float:', als12_ts13_float)

        # print ('---1---')
        # print ('candi:', sc_patient_candi)
        if 'hunt12' in sc_patient_candi:
            # print ('---2---')
            # print ('path_check_h:', os.path.join(path_dir_pkl, sc_patient_candi))
            with open(os.path.join(path_dir_pkl, sc_patient_candi),'rb') as f:
                hunt12_dic = pickle.load(f)
            hunt12_time_left = hunt12_dic['time_left']
            hunt12_time_right = hunt12_dic['time_right']

            hunt12_left_foot = hunt12_dic['left_data']
            hunt12_right_foot = hunt12_dic['right_data']

            hunt12_ts13 = hunt12_dic['ts13_array']
            hunt12_ts13_float = ndstrarr2ndarray(hunt12_ts13)
            #
            ts_100_indice = [5,6,9,10,12]
            for k in ts_100_indice:
                hunt12_ts13_float[:,k] = hunt12_ts13_float[:,k] / 100

    return als12_time_left, als12_time_right, als12_left_foot, als12_right_foot,als12_ts13,    hunt12_time_left, hunt12_time_right, hunt12_left_foot, hunt12_right_foot, hunt12_ts13


def vis_gait64_als(als12):

    plt.figure(figsize=(15,8))

    # left-bottom
    plt.subplot(2,2,3)
    time_left = als12[0]
    time_right = als12[1]
    left_foot = als12[2]
    right_foot = als12[3]

    plt.plot(time_left, left_foot, label = 'Left Foot',  color = 'green', alpha = 0.7)
    plt.plot(time_right, right_foot, label = 'Right Foot', color = 'blue', alpha=0.7)
    plt.xlabel('Time(z)')
    plt.ylabel('Foot Pressure(GRF)')
    plt.title('ALS Patient 12: Raw Foot Pressure Signals')
    plt.legend()
    # 
    plt.grid(True, alpha=0.3)

    # plt.show()
    plt.savefig(r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_board\als12_GRF.jpg', dpi=300)

    # plt.clf()
    plt.cla()
    # plt.close()

    # 
    plt.subplot(1,1,1)
    zoom_end = min(5, len(time_left))
    plt.plot(time_left[:zoom_end], left_foot[:zoom_end], label = 'left Foot[-5th]', color = 'red')
    plt.plot(time_right[:zoom_end], right_foot[:zoom_end], label = 'right Foot[-5th]', color = 'magenta')
    plt.xlabel('Time(z)')
    plt.ylabel('First 5 Seconds: Detailed View')
    plt.legend()
    plt.savefig(r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_board\als12_GRF_5th.jpg', dpi=300)
    plt.cla()




def vis_gait64_hunt(hunt12):

    plt.figure(figsize=(15,8))

    # left-bottom
    time_left = hunt12[0]
    time_right = hunt12[1]
    left_foot = hunt12[2]
    right_foot = hunt12[3]
    #
    plt.subplot(2,2,3)
    plt.plot(time_left, left_foot, label = 'Left Foot',  color = 'green', alpha = 0.7)
    plt.plot(time_right, right_foot, label = 'Right Foot', color = 'blue', alpha=0.7)
    plt.xlabel('Time(z)')
    plt.ylabel('Foot Pressure(GRF)')
    plt.title('HUNT Patient 12: Raw Foot Pressure Signals')
    plt.legend()
    # 
    plt.grid(True, alpha=0.3)

    # plt.show()
    plt.savefig(r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_board\hunt12_GRF.jpg', dpi=300)
    # plt.clf()
    plt.cla()
    # plt.close()

    # 
    plt.subplot(1,1,1)
    zoom_end = min(5, len(time_left))
    plt.plot(time_left[:zoom_end], left_foot[:zoom_end], label = 'Left Foot(-5th)', color = 'red')
    plt.plot(time_right[:zoom_end], right_foot[:zoom_end], label = 'Right Foot[-5th]', color = 'magenta')
    plt.xlabel('Time(z)')
    plt.ylabel('First 5 Seconds: Detailed View')
    plt.legend()
    plt.savefig(r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_board\hunt12_GRF_5th.jpg', dpi=300)
    plt.close()



# 
def vis_gait64_als_ts(als12):


    plt.subplot(4,4,(3,14))


    colors_candidates = ('orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'snow', 'crimson', 'firebrick', 'darkgreen', 'gray', 'k')

    # labelList
    # lables_candidates = ['Elapsed Time (sec)','Left Stride Interval (sec) ',
    # 'Right Stride Interval (sec) ','Left Swing Interval (sec)',
    # 'Right Swing Interval (sec)','Left Swing Interval (% of stride)',
    # 'Right Swing Interval (% of stride)','Left Stance Interval (sec)',
    # 'Right Stance Interval (sec)','Left Stance Interval (% of stride)',
    # 'Double Support Interval (sec)','Double Support Interval (% of stride)']

    lables_candidates = ['Left Stride Interval (sec) ',
    'Right Stride Interval (sec) ','Left Swing Interval (sec)',
    'Right Swing Interval (sec)','Left Swing Interval (% of stride)',
    'Right Swing Interval (% of stride)','Left Stance Interval (sec)',
    'Right Stance Interval (sec)','Left Stance Interval (% of stride)',
    'Right Stance Interval (% of stride)','Double Support Interval (sec)',
    'Double Support Interval (% of stride)']

    # First, select the column, and then read it in a loop.
    als12_ts13 = als12[4]
    # 
    for i in range(als12_ts13.shape[1]-1):
        plt.plot(np.arange(0, als12_ts13.shape[0]), als12_ts13[:, (i+1)], label = lables_candidates[i],  color = colors_candidates[i])

    plt.xlabel('The Length of Time Series for ALS')
    plt.ylabel('Stride/Swing/Stance interval sec or % of stride for ALS')
    plt.legend()
    plt.savefig(r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_board\gait64_degen_ts4als.jpg', dpi=300)
    plt.cla()




def vis_gait64_hunt_ts(hunt12):

    plt.subplot(4,4,(3,14))
    colors_candidates = ('orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'snow', 'crimson', 'firebrick', 'darkgreen', 'gray', 'k')
    lables_candidates = ['Left Stride Interval (sec) ',
    'Right Stride Interval (sec) ','Left Swing Interval (sec)',
    'Right Swing Interval (sec)','Left Swing Interval (% of stride)',
    'Right Swing Interval (% of stride)','Left Stance Interval (sec)',
    'Right Stance Interval (sec)','Left Stance Interval (% of stride)',
    'Right Stance Interval (% of stride)','Double Support Interval (sec)',
    'Double Support Interval (% of stride)']
    hunt12_ts13 = hunt12[4]
    # 
    for i in range(hunt12_ts13.shape[1] - 1):
        plt.plot(np.arange(0, hunt12_ts13.shape[0]), hunt12_ts13[:, (i+1)], label = lables_candidates[i],  color = colors_candidates[i])
    plt.xlabel('The Length of Time Series for HUNT')
    plt.ylabel('Stride/Swing/Stance interval sec or % of stride for HUNT')
    plt.legend()
    plt.savefig(r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_board\gait64_degen_ts4hunt.jpg', dpi=300)
    plt.cla()


if __name__=="__main__":
    #
    path_pkl = "D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\gait64_neodegen_pkls"

    # 
    '''
    '''
    als12_time_left, als12_time_right, als12_left_foot, als12_right_foot, als12_ts13,     hunt12_time_left, hunt12_time_right, hunt12_left_foot, hunt12_right_foot, als12_ts13  = load_pkl_gait64_two_cases(path_pkl)

    # 
    als12 = (als12_time_left, als12_time_right, als12_left_foot, als12_right_foot, als12_ts13)
    hunt12 = (hunt12_time_left, hunt12_time_right, hunt12_left_foot, hunt12_right_foot, als12_ts13)

    #
    vis_gait64_als(als12)
    vis_gait64_hunt(hunt12)

    vis_gait64_als_ts(als12)
    vis_gait64_hunt_ts(als12)
