import time
import signal
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from helper import set_board_channel_settings, set_board_timestamp
import struct
import asyncio
import socketio
sio = socketio.AsyncClient()
ws_url = 'http://127.0.0.1:8080'

@sio.event
async def connect():
    print('connected to server')

@sio.event
async def disconnect():
    print('disconnected from server')

def prepOpenBCIBoard():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    # params.serial_port = '/dev/ttyUSB0'
    params.serial_port = 'COM9'
    # params.serial_port = '/dev/ttyDUMMY'
    cyton_id = BoardIds['CYTON_BOARD']
    global board_shim
    global sample_rate
    global exg_channels
    global accel_channels
    global timestamp_s_channel
    board_shim = BoardShim(cyton_id, params)
    timestamp_s_channel = [15, 16, 17, 18]
    sample_rate = board_shim.get_sampling_rate(cyton_id)
    exg_channels = board_shim.get_exg_channels(cyton_id)
    accel_channels = board_shim.get_accel_channels(cyton_id)
    num_channel = board_shim.get_package_num_channel(cyton_id)

    # print(timestamp_channel)
    # print(exg_channels)
    # print(accel_channels)
    # print(exg_channels + accel_channels + [timestamp_channel])

    # exit()
    board_shim.prepare_session()
    for channel_i in range(1, len(exg_channels)):
        set_board_channel_settings(board_shim, channel_i)
    set_board_timestamp(board_shim, True)


async def brainFlowStream():
    global board_shim
    global sample_rate
    global exg_channels
    global accel_channels
    global timestamp_s_channel
    sample_interval = 1 / sample_rate
    sample_interval_e_margin = sample_interval * 0.5
    board_shim.start_stream()
    # prev_buff_time = time.time()
    # for i in range(1000000000):
    #     data = np.array(board_shim.get_board_data())
    #     if data.size > 1:
    #         timestamp = data[timestamp_channel, 0]
    #         print(prev_time, timestamp)
    #         print(prev_time - timestamp)
    #         prev_time = timestamp
    while True:
        data = np.array(board_shim.get_board_data())
        if data.size > 0:

            # curr_buff_time = data[timestamp_channel, -1]
            # print(data[10:22].T)
            # print(time.time(), curr_buff_time)
            # num_frame = data.shape[1]
            # too_far_back = False
            # for frame_i in range(num_frame):
            #     if too_far_back:
            #         frame_i_time = prev_buff_time
            #     else :
            #         frame_i_time = curr_buff_time - (num_frame - 1 - frame_i) * sample_interval
            #         too_far_back = frame_i_time < prev_buff_time
            #     data[timestamp_channel, frame_i] = frame_i_time
            # prev_buff_time = curr_buff_time

            for frame_i in range(data.shape[1]):
                timestamp_s_raw = np.array(data[timestamp_s_channel, frame_i], dtype=np.uint8)
                timestamp_s = (struct.unpack('>I', struct.pack('BBBB', *timestamp_s_raw))[0]) / 20000
                print(f'Got data at device time: {timestamp_s:.5f}s')
                EXG_data = data[exg_channels, frame_i]
                # data_frame = np.zeros(len(EXG_data) + 1)
                # data_frame[:-1] = EXG_data
                # data_frame[-1] = timestamp
                if sio.connected:
                    await sio.emit('EEG_data', struct.pack('d' * len(EXG_data) + 'd', *EXG_data, timestamp_s))
        await asyncio.sleep(0)

async def ws_connect():
    await sio.connect(ws_url)
    await sio.wait()

async def main():
    prepOpenBCIBoard()
    asyncio.ensure_future(brainFlowStream())
    await ws_connect()

def exit_handler(signum, frame):
    print('ctrl-c received! Closing session and exiting')
    global board_shim
    set_board_timestamp(board_shim, False)
    board_shim.stop_stream()
    board_shim.release_session()
    exit()

signal.signal(signal.SIGINT, exit_handler)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())