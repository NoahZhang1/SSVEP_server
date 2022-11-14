from aiohttp import web
import socketio
import struct
import time
import datetime
import numpy as np
import msgpack
import msgpack_numpy as m
import scipy.signal
import atexit
# import rcca
from sklearn.cross_decomposition import CCA
import os

m.patch()
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

now = datetime.datetime.now()
DATA_STORAGE_PATH = f'./sweep_data_{now.strftime("%m%d%Y_%H%M")}'
NUM_STIMULUS = 40
NUM_EEG_CHNL = 8
EEG_SAMPLE_RATE_Hz = 250
EEG_SAMPLE_INTERVAL_S = 1 / EEG_SAMPLE_RATE_Hz
STIMULI_SAMPLE_RATE_Hz = 60
CRITICAL_SEGMENT_S = 5
BLINKING_TIME_S = 5.5
TIME_BEFORE_START_BLINKING_TIME_S = 2.5
BLINKING_TIME_NUM_FRAME = round(
    (BLINKING_TIME_S + TIME_BEFORE_START_BLINKING_TIME_S) * EEG_SAMPLE_RATE_Hz
)
CRITICAL_SEGMENT_SIZE = round(EEG_SAMPLE_RATE_Hz * CRITICAL_SEGMENT_S)
BLINKING_NUM_FRAME_ERR_MARGIN = round(BLINKING_TIME_NUM_FRAME * 0.03)
EEG_DATA_BUFFER_TIME_S = BLINKING_TIME_S + 5
EEG_DATA_BUFFER_SIZE = round(EEG_SAMPLE_RATE_Hz * EEG_DATA_BUFFER_TIME_S)
STIMULI_BUFFER_SIZE = round(STIMULI_SAMPLE_RATE_Hz * EEG_DATA_BUFFER_TIME_S)
NUM_HARMONICS = 4

stimulus_timestamp = np.zeros(STIMULI_BUFFER_SIZE, dtype=np.float64)
stimulus_local_timestamp = np.zeros(STIMULI_BUFFER_SIZE, dtype=np.float64)
stimuli_phase_buffer = np.zeros(
    (NUM_STIMULUS, STIMULI_BUFFER_SIZE), dtype=np.float64
)

EEG_data_buffer = np.zeros(
    (NUM_EEG_CHNL, EEG_DATA_BUFFER_SIZE), dtype=np.float64
)
EEG_data_timestamp = np.zeros(EEG_DATA_BUFFER_SIZE, dtype=np.float64)
EEG_data_local_timestamp = np.zeros(EEG_DATA_BUFFER_SIZE, dtype=np.float64)
EEG_buffer_curser = 0
stimuli_phase_buffer_curser = 0
# EEG_timestamp_to_local_timestamp_mod = LinearRegression()
start_blinking_local_time = 0
stop_blinking_local_time = 0
start_blinking_web_time = 0
stop_blinking_web_time = 0

# filterbank_freq_ranges = [(8, 88), (16, 88), (24, 88), (32, 88), (40, 88)]
filterbank_freq_ranges = [(8, 88)]


def bandpass(lowcut, highcut, fs, filter_type='butter', order=4, rp=0.1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if filter_type == 'butter':
        sos = scipy.signal.butter(
            order, [low, high], analog=False, btype='band', output='sos'
        )
    elif filter_type == 'cheby1':
        sos = scipy.signal.cheby1(
            order, rp, [low, high], analog=False, btype='band', output='sos'
        )
    # elif filter_type == 'cheby2':
    # sos = scipy.signal.cheby2(
    #     order, rp, [low, high], analog=False, btype='band', output='sos'
    # )
    return sos


filter_params = [
    bandpass(low, high, EEG_SAMPLE_RATE_Hz) for low, high in
    filterbank_freq_ranges
]
stimulus_freq_hz = np.array([
    [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    [8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2],
    [8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4],
    [8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6],
    [8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8]
], dtype=np.float64)

stimulus_phase_offset = np.array([
    [0.00, 1.75, 1.50, 1.25, 1.00, 0.75, 0.50, 0.25],
    [0.35, 0.10, 1.85, 1.60, 1.35, 1.10, 0.85, 0.60],
    [0.70, 0.45, 0.20, 1.95, 1.70, 1.45, 1.20, 0.95],
    [1.05, 0.80, 0.55, 0.30, 0.05, 1.80, 1.55, 1.30],
    [1.40, 1.15, 0.90, 0.65, 0.40, 0.15, 1.90, 1.55]
], dtype=np.float64) * np.pi


@sio.event
def connect(sid, environ):
    EEG_buffer_curser = 0
    stimuli_phase_buffer_curser = 0
    print(f'client {sid} connected')


@sio.event
def disconnect(sid):
    print('Client disconnected')


@sio.event
def phase_update(sid, message):
    local_timestamp_s = time.time()
    update_data = struct.unpack('d' * NUM_STIMULUS + 'd', message)
    phase_data = np.array(update_data[:-1], dtype=np.float64)
    timestamp_s = update_data[-1]
    global stimuli_phase_buffer_curser
    if stimuli_phase_buffer_curser < STIMULI_BUFFER_SIZE:
        stimuli_phase_buffer[:, stimuli_phase_buffer_curser] = phase_data
        stimulus_timestamp[stimuli_phase_buffer_curser] = timestamp_s
        stimulus_local_timestamp[stimuli_phase_buffer_curser] = local_timestamp_s
        stimuli_phase_buffer_curser = stimuli_phase_buffer_curser + 1
    else:
        stimuli_phase_buffer[:, 0:-1] = stimuli_phase_buffer[:, 1:]
        stimuli_phase_buffer[:, -1] = phase_data
        stimulus_timestamp[0:-1] = stimulus_timestamp[1:]
        stimulus_timestamp[-1] = timestamp_s
        stimulus_local_timestamp[0:-1] = stimulus_local_timestamp[1:]
        stimulus_local_timestamp[-1] = local_timestamp_s

    global stim_phase_fh
    data_buf = struct.pack(
        'd' * NUM_STIMULUS + 'dd', *update_data, local_timestamp_s
    )
    if not stim_phase_fh.closed:
        stim_phase_fh.write(data_buf)


TEST_freq_Hz = 0


@sio.event
def freq_update_all(sid, message):
    local_timestamp_s = time.time()
    global TEST_freq_Hz
    previous_freq = TEST_freq_Hz
    TEST_freq_Hz, freq_update_all_web_timestamp = struct.unpack('dd', message)
    print(f'stim: {TEST_freq_Hz:.1f}Hz web_timestamp: {freq_update_all_web_timestamp:.2f}s')
    global stim_event_fh
    data_buf = struct.pack(
        'ddd', TEST_freq_Hz, freq_update_all_web_timestamp, local_timestamp_s
    )
    if (not stim_event_fh.closed) and (TEST_freq_Hz > 0):
        stim_event_fh.write(data_buf)
    if (TEST_freq_Hz == 0) and (previous_freq > 0):
        close_data_files()


@sio.event
def is_blinking(sid, message):
    local_timestamp_s = time.time()
    is_blinking, timestamp_s = struct.unpack('Qd', message)
    global start_blinking_local_time
    global start_blinking_web_time
    global stop_blinking_local_time
    global stop_blinking_web_time
    global TEST_freq_Hz

    if is_blinking:
        start_blinking_local_time = local_timestamp_s
        start_blinking_web_time = timestamp_s

    elif not (EEG_buffer_curser < EEG_DATA_BUFFER_SIZE):
        stop_blinking_local_time = local_timestamp_s
        stop_blinking_web_time = timestamp_s
        # blinking_data_mask = np.logical_and(EEG_data_local_timestamp <= stop_blinking_local_time, EEG_data_local_timestamp >= start_blinking_local_time)

        blinking_data_mask = np.logical_and(EEG_data_local_timestamp <= stop_blinking_local_time, EEG_data_local_timestamp >= (start_blinking_local_time - TIME_BEFORE_START_BLINKING_TIME_S))
        num_blinking_frame = np.sum(blinking_data_mask)
        print(f'{num_blinking_frame} / {BLINKING_TIME_NUM_FRAME}')
        if (num_blinking_frame > (BLINKING_TIME_NUM_FRAME - BLINKING_NUM_FRAME_ERR_MARGIN)) and (num_blinking_frame < (BLINKING_TIME_NUM_FRAME + BLINKING_NUM_FRAME_ERR_MARGIN)):
            print(f'GOT data!!!! {TEST_freq_Hz}Hz {num_blinking_frame} / {BLINKING_TIME_NUM_FRAME}')
            EEG_data_to_process = EEG_data_buffer[:, blinking_data_mask]
            EEG_data_timestamp_to_process = EEG_data_timestamp[blinking_data_mask]
            EEG_data_local_timestamp_to_process = EEG_data_local_timestamp[blinking_data_mask]
            # time_arr = np.arange(start_blinking_web_time, start_blinking_web_time + EEG_SAMPLE_INTERVAL_S * num_blinking_frame ,EEG_SAMPLE_INTERVAL_S)[:num_blinking_frame]
            # reference_signals = np.vstack([generate_ref_signal(time_arr, i)[np.newaxis, ...] for i in range(NUM_HARMONICS)])
            # process_data(EEG_data_to_process, reference_signals)
            stimuli_mask = np.logical_and(stimulus_local_timestamp <= stop_blinking_local_time, stimulus_local_timestamp >= (start_blinking_local_time - TIME_BEFORE_START_BLINKING_TIME_S))
            stimuli_phase_buffer_to_process = stimuli_phase_buffer[:, stimuli_mask]
            stimulus_local_timestamp_to_process = stimulus_local_timestamp[stimuli_mask]
            stimulus_timestamp_to_process = stimulus_timestamp[stimuli_mask]
            # if (TEST_freq_Hz > 0.0):
            #     print(f'{DATA_STORAGE_PATH}/debug_{TEST_freq_Hz:.1f}hz_0phase_data.bin')
            #     with open(f'{DATA_STORAGE_PATH}/debug_{TEST_freq_Hz:.1f}hz_0phase_data.bin', 'wb') as fh:
            #         fh.write(msgpack.packb({
            #             'eeg_data': EEG_data_to_process,
            #             'EEG_data_timestamp': EEG_data_timestamp_to_process,
            #             'EEG_data_local_timestamp':EEG_data_local_timestamp_to_process,
            #             'start_blinking_web_time': start_blinking_web_time,
            #             'start_blinking_local_time': start_blinking_local_time,
            #             'stop_blinking_local_time': stop_blinking_local_time,
            #             'stop_blinking_web_time': stop_blinking_web_time,
            #             'stimuli_phase_buffer': stimuli_phase_buffer_to_process,
            #             'stimulus_local_timestamp': stimulus_local_timestamp_to_process,
            #             'stimulus_timestamp': stimulus_timestamp_to_process
            #         }))


def generate_ref_signal(time_arr, harmonic=1, visual_delay_s=0.140):
    reference_signals = np.zeros(
        stimulus_freq_hz.shape + (time_arr.size,), dtype=np.float64
    )
    for row in range(stimulus_freq_hz.shape[0]):
        for col in range(stimulus_freq_hz.shape[1]):
            f_Hz = stimulus_freq_hz[row][col]
            phase_rad = stimulus_phase_offset[row][col]
            reference_signals[row, col] = np.cos(
                harmonic * (2 * np.pi * f_Hz * (
                    time_arr + visual_delay_s
                ) + phase_rad)
            )
    return reference_signals


def process_data(EEG_data_to_process, reference_signals):
    filtered_EEG_data = np.zeros(
        (len(filter_params),) + EEG_data_to_process.shape, dtype=np.float64
    )
    for i, filter_param in enumerate(filter_params):
        filtered_EEG_data[i] = scipy.signal.sosfiltfilt(
            filter_param, EEG_data_to_process
        )
    reference_signals_flatten = reference_signals.reshape(
        (np.product(reference_signals.shape[0:3]), reference_signals.shape[-1])
    )

    offset = round(
        (reference_signals_flatten.shape[1] - CRITICAL_SEGMENT_SIZE) / 2
    )
    reference_signals_flatten = reference_signals_flatten[
        :, offset:offset+CRITICAL_SEGMENT_SIZE
    ]
    filtered_EEG_data = filtered_EEG_data[
        :, :, offset:offset+CRITICAL_SEGMENT_SIZE
    ]
    # print(filtered_EEG_data.shape, reference_signals_flatten.shape)
    for filtered_EEG_sb in filtered_EEG_data:
        ref_signal_weights = cca_analysis(
            filtered_EEG_sb, reference_signals_flatten
        )[:-1]
        ref_signal_weights = ref_signal_weights.reshape(
            (NUM_HARMONICS,) + stimulus_freq_hz.shape
        )
        # print(ref_signal_weights.shape)
        # print(np.sum(np.sum(ref_signal_weights, axis=-1), axis=-1))
        print('---------------------------------------------------------------'
              '-----------------'
        )
        for h_i in range(NUM_HARMONICS):
            ref_signal_weights_hi = ref_signal_weights[h_i]
            max_row_col = np.argwhere(
                ref_signal_weights_hi == np.max(ref_signal_weights_hi)
            )[0]
            print(f'h{h_i} w:{np.max(ref_signal_weights_hi):.2f} row:{max_row_col[0]} col:{max_row_col[1]} f:{stimulus_freq_hz[max_row_col[0], max_row_col[1]]} phase:{ stimulus_phase_offset[max_row_col[0], max_row_col[1]]}')

    # print(filtered_EEG_data.shape)
    # print(EEG_data_to_process.shape)
    # print(reference_signals.shape)


def cca_analysis(EEG_data_subband, reference_signals):
    cca = CCA(n_components=1)
    reference_signals = np.vstack(
        (reference_signals, np.ones(reference_signals.shape[1]))
    )
    cca.fit(EEG_data_subband.T, reference_signals.T)
    return cca.y_weights_


@sio.event
def EEG_data(sid, message):
    now_time = time.time()
    eeg_data_msg = struct.unpack('d' * (NUM_EEG_CHNL) + 'd', message)
    timestamp_ms = eeg_data_msg[-1]
    # print(f'Got EEG data with timestamp {timestamp_ms}')
    eeg_data = np.array(eeg_data_msg[:-1], dtype=np.float64)
    global EEG_buffer_curser
    local_timestamp_s = now_time

    # if (EEG_buffer_curser == 0):
        # local_timestamp_s = now_time
    # else:
        # timestamp_delta_s = (timestamp_ms - EEG_data_timestamp[EEG_buffer_curser-1]) / 1e3
        # local_timestamp_s = EEG_data_local_timestamp[EEG_buffer_curser-1] + timestamp_delta_s

    if EEG_buffer_curser < EEG_DATA_BUFFER_SIZE:
        EEG_data_buffer[:, EEG_buffer_curser] = eeg_data
        EEG_data_timestamp[EEG_buffer_curser] = timestamp_ms
        EEG_data_local_timestamp[EEG_buffer_curser] = local_timestamp_s
        EEG_buffer_curser = EEG_buffer_curser + 1
    else:
        EEG_data_buffer[:, 0:-1] = EEG_data_buffer[:, 1:]
        EEG_data_buffer[:, -1] = eeg_data
        EEG_data_timestamp[0:-1] = EEG_data_timestamp[1:]
        EEG_data_timestamp[-1] = timestamp_ms
        EEG_data_local_timestamp[0:-1] = EEG_data_local_timestamp[1:]
        EEG_data_local_timestamp[-1] = local_timestamp_s

    global EEG_data_fh
    data_buf = struct.pack(
        'd' * (NUM_EEG_CHNL) + 'dd', *eeg_data_msg, now_time
    )
    if not EEG_data_fh.closed:
        EEG_data_fh.write(data_buf)


def init_app():
    return app


@atexit.register
def exit_handler():
    close_data_files()


def close_data_files():
    global EEG_data_fh
    if not EEG_data_fh.closed:
        EEG_data_fh.close()

    global stim_phase_fh
    if not stim_phase_fh.closed:
        stim_phase_fh.close()

    global stim_event_fh
    if not stim_event_fh.closed:
        stim_event_fh.close()


def main():
    if not os.path.exists(DATA_STORAGE_PATH):
        os.mkdir(DATA_STORAGE_PATH)
    global EEG_data_fh
    EEG_data_fh = open(f'{DATA_STORAGE_PATH}/EEG_data.bin', 'wb')
    global stim_phase_fh
    stim_phase_fh = open(f'{DATA_STORAGE_PATH}/stim_phase.bin', 'wb')
    global stim_event_fh
    stim_event_fh = open(f'{DATA_STORAGE_PATH}/stim_event.bin', 'wb')
    web.run_app(init_app(), host='127.0.0.1', port=8080)


if __name__ == '__main__':
    main()