import scipy.signal
import numpy as np


def bandpass(lowcut, highcut, fs, filter_type='butter', order=4, rp=0.1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if filter_type == 'butter':
        sos = scipy.signal.butter(
            order, [low, high], analog=False, btype='band', output='sos')
    elif filter_type == 'cheby1':
        sos = scipy.signal.cheby1(
          order, rp, [low, high], analog=False, btype='band', output='sos')
    # elif filter_type == 'cheby2':
        # sos = scipy.signal.cheby2(
        #    order, rp, [low, high], analog=False, btype='band', output='sos')
    return sos


def rad_to_deg(rad):
    return rad / (2 * np.pi) * 360


def find_closest_ndx_in_sorted_arr(arr, num):
    arr = np.array(arr)
    i_small = np.sum(arr < num)
    i_large = i_small + 1
    diff_with_small = num - arr[i_small]
    diff_with_large = arr[i_large] - num
    i_min = i_small
    if diff_with_large < diff_with_small:
        i_min = i_large
    return i_min



def set_board_channel_settings(
    board_shim, channel, POWER_DOWN=0, GAIN_SET=6, INPUT_TYPE_SET=0,
    BIAS_SET=0, SRB2_SET=1, SRB1_SET=0
):
    return board_shim.config_board(
      f'x{channel}{POWER_DOWN}{GAIN_SET}{INPUT_TYPE_SET}{BIAS_SET}{SRB2_SET}'
      f'{SRB1_SET}X'
    )


def set_board_timestamp(board_shim, timestamp_on):
    if timestamp_on:
        return board_shim.config_board('<')
    else:
        return board_shim.config_board('>')


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def web_timestamp_to_phase(
    web_timestamp, freq_Hz, visual_phase_delay_s=0, phase_offset=0, harmonic=1
):
    return 2 * np.pi * (freq_Hz * harmonic) * \
        (web_timestamp + visual_phase_delay_s) + \
        phase_offset  + (harmonic - 1) * np.pi
