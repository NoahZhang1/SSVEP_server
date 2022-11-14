from unittest import result
from helper import bandpass, rad_to_deg, find_closest_ndx_in_sorted_arr
import numpy as np
import msgpack
import msgpack_numpy as m
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.signal
import scipy.ndimage
import math
import os
import struct
import warnings
import heapq
from scipy.stats import pearsonr
from scipy import spatial
warnings.filterwarnings("ignore", category=DeprecationWarning) 
plt.style.use('seaborn')
m.patch()

# folder_path = './sweep_data_10072022_1718'
folder_path = './sweep_data_10112022_2237'
# folder_path = './sweep_data_10112022_2248'
# folder_path = './sweep_data_10112022_2303'
# folder_path = './sweep_data_10112022_2318'

PHOTODIODE_DATA_LIST = [
    './sweep_data_10112022_2237',
    './sweep_data_10112022_2248',
    './sweep_data_10112022_2303',
    './sweep_data_10112022_2318'
]

HAS_PHOTODIODE = folder_path in PHOTODIODE_DATA_LIST

EEG_data_path = f'{folder_path}/EEG_data.bin'
stim_event_path = f'{folder_path}/stim_event.bin'
stim_phase_path = f'{folder_path}/stim_phase.bin'

EEG_SAMPLE_RATE_Hz = 250
EEG_SAMPLE_INTERVAL_S = 1 / EEG_SAMPLE_RATE_Hz
NUM_CHNL = 8
monitor_refresh_rate_Hz = 60
phase_offset_rad = 0


CRITICAL_SEGMENT_S = 5.5

NUM_STIMULUS = 40
INSPECT_SEGMENT_S = 1
OFFSET_TIME_S = 0.07

REACTION_FRAME = int(OFFSET_TIME_S * EEG_SAMPLE_RATE_Hz)
INSPECT_FRAME = int(INSPECT_SEGMENT_S * EEG_SAMPLE_RATE_Hz)
win_len = INSPECT_FRAME
normalize_factor = 2 / win_len
xf = fftfreq(win_len, 1 / EEG_SAMPLE_RATE_Hz)
xf_mask = np.logical_and(xf > 6, xf <= 80)
xf_masked = xf[xf_mask]
normalize_factor = 2 / win_len

# propogation_delay_s = 0.148

# magic_stim_offset account for phaseshift due to down sampling to 60Hz
magic_stim_offset = -(1 / monitor_refresh_rate_Hz) / 2


# cumulated communication delay, monitor display delay, and other system delay
#default
magic_data_system_delay_s = 0.160

# magic_data_system_delay_s = 0

#8hz = 0.125 seconds
#20hz = 0.05 second
with open(EEG_data_path, 'rb') as fh:
    EEG_data_parsed = list(struct.iter_unpack('dddddddddd', fh.read()))
EEG_data = np.array([entry[:8] for entry in EEG_data_parsed])
EEG_device_timestamp = np.array([entry[8] for entry in EEG_data_parsed])
EEG_pc_timestamp = np.array([entry[9] for entry in EEG_data_parsed])
# EEG_pc_timestamp -= magic_data_system_delay_s

'''This section reads the stimuli event

stim_event_parsed: raw parsed data read from file
stim_event: np array, first entry of stim_event_parsed

'''
with open(stim_event_path, 'rb') as fh:
    stim_event_parsed = list(struct.iter_unpack('ddd', fh.read()))
stim_event = np.array([entry[0] for entry in stim_event_parsed])
stim_web_timestamp = np.array([entry[1] for entry in stim_event_parsed])
stim_pc_timestamp = np.array([entry[2] for entry in stim_event_parsed])

'''This section reads the actual 8-channel data

    EEG_data_parsed: raw data
    EEG_data: np array, first eight entries from eeg_data
    stim_event: np array, first entry of stim_event_parsed

    '''
'frequencies: np array, frequencies of stimulus'
frequencies = stim_event

lowcut, highcut = np.min(frequencies) - 4, np.max(frequencies) * 4 + 8
filter_param = bandpass(lowcut, highcut, EEG_SAMPLE_RATE_Hz, order=6)

with open(stim_phase_path, 'rb') as fh:
    stim_phase_parsed = list(
        struct.iter_unpack('d'*NUM_STIMULUS+'dd', fh.read())
    )
phase = np.array([entry[0] for entry in stim_phase_parsed])
phase_web_timestamp = np.array([
    entry[NUM_STIMULUS] for entry in stim_phase_parsed
])
phase_pc_timestamp = np.array([
    entry[NUM_STIMULUS + 1] for entry in stim_phase_parsed
])

inspection_EEG_segment = []
inspection_EEG_segment_index = []
inspection_web_timestamp = []
inspection_stim_phase = []

stim_pc_timestamp = np.array([
    phase_pc_timestamp[phase_pc_timestamp >= stim_pc_ts][0]
    for stim_pc_ts in stim_pc_timestamp
])

stim_web_timestamp = np.array([
    phase_web_timestamp[phase_web_timestamp >= stim_web_ts][0]
    for stim_web_ts in stim_web_timestamp
])

# time_correction_s = phase_update_start_pc_timestamp - stim_pc_timestamp
# stim_web_timestamp = stim_web_timestamp + time_correction_s
# array_1
# array_2
# zip(range(len(array_1)),)

for stim_event_hz, web_timestamp_s, pc_timestamp_s in zip(
    stim_event, stim_web_timestamp, stim_pc_timestamp
):
    eeg_start_i = find_closest_ndx_in_sorted_arr(
        EEG_pc_timestamp, pc_timestamp_s
    )
    # print(web_timestamp_s)
    eeg_inspection_start_i = eeg_start_i + REACTION_FRAME
    eeg_inspection_end_i = eeg_inspection_start_i + INSPECT_FRAME
    inspection_EEG_segment_index.append(
        (eeg_inspection_start_i, eeg_inspection_end_i)
    )
    inspection_EEG_segment.append(
        EEG_data[eeg_inspection_start_i:eeg_inspection_end_i, :]
    )
    event_start = web_timestamp_s + OFFSET_TIME_S
    event_end = web_timestamp_s + OFFSET_TIME_S + INSPECT_SEGMENT_S
    event_timestamp = np.linspace(
        event_start, event_end, INSPECT_FRAME, endpoint=False
    )
    inspection_web_timestamp.append(event_timestamp)
    stim_phase_mask = np.logical_and(
        phase_web_timestamp >= event_start,
        phase_web_timestamp <= event_end
    )
    stim_event_phase = phase[stim_phase_mask]
    stim_event_web_timestamp = phase_web_timestamp[stim_phase_mask]
    # stim_event_phase_dense_timestamp = np.copy(event_timestamp)
    dense_phase = np.zeros(event_timestamp.shape)
    for i in range(len(stim_event_phase) - 1):
        mask_i = np.logical_and(
                event_timestamp >= stim_event_web_timestamp[i],
                event_timestamp < stim_event_web_timestamp[i+1]
            )
        dense_phase[mask_i] = stim_event_phase[i]
    dense_phase_stim_vale = np.cos(dense_phase)
    inspection_stim_phase.append(dense_phase_stim_vale)


response_dict = {}

for freq_i, freq_hz in enumerate(frequencies):
    period_s = 1 / freq_hz
    # eeg_seg = inspection_EEG_segment[freq_i]
    eeg_seg_start, eeg_seg_end = inspection_EEG_segment_index[freq_i]
    web_timestamp_seg = inspection_web_timestamp[freq_i]
    phase_update_response = inspection_stim_phase[freq_i]
    

    lowcut_h1, highcut_h1 = 1 * freq_hz - 0.5, 1 * freq_hz + 0.5

    filter_param_h1 = bandpass(
        lowcut_h1, highcut_h1, EEG_SAMPLE_RATE_Hz, order=8
    )

    filtered_eeg_h1 = scipy.signal.sosfiltfilt(
        filter_param_h1, EEG_data.T).T[eeg_seg_start: eeg_seg_end, :]

    #This is the expected data:
    # expected_response_list = []

    #This is the observed data:
    response_list = []


    for channel_num in range(NUM_CHNL):
        eeg_data = filtered_eeg_h1[:, channel_num]
        response_mag = np.max(eeg_data)
        response_list.append(eeg_data)
        # print("speed test")
        # plt.plot(expected_response)
        # plt.show()
    response_dict[round(freq_hz,1)] = response_list



result_array = {}
selected_frequency = 9.2
# start_search_val = -0.4
end_search_val = -0.1
start_search_val = end_search_val - (1/selected_frequency)
search_interval = 0.001
for loop, offset in enumerate(np.arange(start_search_val, end_search_val, search_interval)):
    

    # variation = loop/10000


    # print(loop)
    # print(offset)
    
    # OFFSET_TIME_S = 0.15


    # default()
    VISUAL_PHASE_OFFSET_S = offset
    print(f'calculating for the {loop+1} iteration, the phase offset is now {VISUAL_PHASE_OFFSET_S}')


    inspection_web_stim_h1 = np.array([
        np.cos(2 * np.pi * freq_hz * (
            + magic_stim_offset + EEG_web_timestamp + VISUAL_PHASE_OFFSET_S
        ) + phase_offset_rad) for freq_hz, EEG_web_timestamp in zip(
            frequencies, inspection_web_timestamp
        )
    ])

    expect_dict = {}
    # response_dict = {}

    for freq_i, freq_hz in enumerate(frequencies):
        # period_s = 1 / freq_hz
        # # eeg_seg = inspection_EEG_segment[freq_i]
        # eeg_seg_start, eeg_seg_end = inspection_EEG_segment_index[freq_i]
        # web_timestamp_seg = inspection_web_timestamp[freq_i]
        # phase_update_response = inspection_stim_phase[freq_i]
        

        # lowcut_h1, highcut_h1 = 1 * freq_hz - 0.5, 1 * freq_hz + 0.5

        # filter_param_h1 = bandpass(
        #     lowcut_h1, highcut_h1, EEG_SAMPLE_RATE_Hz, order=8
        # )

        # filtered_eeg_h1 = scipy.signal.sosfiltfilt(
        #     filter_param_h1, EEG_data.T).T[eeg_seg_start: eeg_seg_end, :]

        #This is the expected data:
        expected_response_list = []

        #This is the observed data:
        # response_list = []


        for channel_num in range(NUM_CHNL):
            # eeg_data = filtered_eeg_h1[:, channel_num]
            expected_response = inspection_web_stim_h1[freq_i]
            # response_mag = np.max(eeg_data)


            expected_response_list.append(response_mag * expected_response)
            # response_list.append(eeg_data)
            # print("speed test")
            # plt.plot(expected_response)
            # plt.show()
        expect_dict[round(freq_hz,1)] = expected_response_list
        # response_dict[round(freq_hz,1)] = response_list
        



    '''Set the frequency to calculate the correlation for this specific delay

    excluding the data from photodiode

    '''
    
    # selected_freq_i = round((selected_frequency-8.0)//0.2)
    # print(f'calculating channel correlation for frequency {selected_frequency}')
    correlation_array = []
    for i in range(1,len(response_dict[selected_frequency])):
        response_list = response_dict[selected_frequency]
        expected_response_list = expect_dict[selected_frequency]
        correlation = pearsonr(expected_response_list[i],response_list[i])
        # correlation = spatial.distance.cosine(response_list[i], expected_response_list[i])
        # print(f'correlation for channel {i+1} is {correlation}')
        correlation_array.append(correlation.statistic)
        # print("correlation arrray is now", correlation_array)
    heapq.heapify(correlation_array)
    heapq.heappop(correlation_array)
    heapq.heappop(correlation_array)
    result_array[offset] = (np.mean(correlation_array))
    
# print(result_array)
max_value = max(result_array.values())
max_key = max(result_array, key = result_array.get)
print(f"the maximum correlation is acheived at {max_key} with value {max_value} for {selected_frequency} Hz")

# print(pearsonr(response_mag * expected_response,response_mag * phase_update_response))
# plt.plot(response_mag * phase_update_response)
# plt.plot(response_mag * expected_response)
# plt.show()
