import asyncio
import struct
from bleak import BleakScanner, BleakClient
import numpy as np
import numba as nb

TARGET_SSID = 'nRF52_EEG'
TARGET_ADDR = 'E4:CE:5D:44:D5:94'
NUS_CHARACTERISTIC_UUID = '6e400003-b5a3-f393-e0a9-e50e24dcca9e'
FIND_DEVICE_TIMEOUT_S = 5.0


def filter_for_target_device(d, ad):
    is_match = False
    if d.address:
        is_match = (TARGET_ADDR == d.address)
    if d.name:
        is_match = is_match and (TARGET_SSID == d.name)
    return is_match


@nb.njit(nb.int16[::1](
    nb.uint8[::1], nb.int16[::1]), fastmath=True, parallel=True, cache=True)
def read_uint12_var_2_prealloc(packed_d, out):
    # https://stackoverflow.com/questions/44735756
    assert np.mod(packed_d.shape[0], 3) == 0
    assert out.shape[0] == packed_d.shape[0] // 3 * 2

    for i in nb.prange(packed_d.shape[0] // 3):
        packed_d_i = i * 3
        out_i = i * 2
        packed_0 = np.uint16(packed_d[packed_d_i + 0])
        packed_1 = np.uint16(packed_d[packed_d_i + 1])
        packed_2 = np.uint16(packed_d[packed_d_i + 2])

        out[out_i] = (packed_0 | (packed_1 << 8)) & 0x0FFF
        out[out_i + 1] = ((packed_1 >> 4) | (packed_2 << 4)) & 0x0FFF
    out[out > 0x0800] |= 0xf000
    return out


def nus_data_handler(sender, data):

    # original_data_size = round(compacted_data_size / 12 * 16)
    # original_data = bytearray(original_data_size)
    compacted_data = np.array(data[:-10], dtype=np.uint8)
    compacted_data_size = len(compacted_data)
    original_data = np.zeros(compacted_data_size // 3 * 2, dtype=np.int16)
    # print(compacted_data.shape)
    read_uint12_var_2_prealloc(compacted_data, original_data)
    # print(original_data[0:8])
    # print(data[0:6])
    # original_data_unpack = list(struct.iter_unpack('<h', original_data))
    time_stamp = int.from_bytes(data[-10:-6], 'little', signed=False)
    # original_data_unpack = np.array(original_data_unpack)[:, 0]
    # print(original_data_unpack)
    print(time_stamp)
    num_frames = int(original_data.size / 8)
    # print(original_data.reshape(num_frames, 8)[0])


async def main():
    while True:
        print('BLE scanning...')
        device = await BleakScanner.find_device_by_filter(
            filter_for_target_device,
            timeout=FIND_DEVICE_TIMEOUT_S
        )
        print(f'Found target device: {device}')
        disconnected_event = asyncio.Event()

        def on_disconnect(client):
            print("on_disconnect")
            disconnected_event.set()
        if device is not None:
            async with BleakClient(
                    device.address, disconnected_callback=on_disconnect
                    ) as client:
                await client.start_notify(
                    NUS_CHARACTERISTIC_UUID, nus_data_handler
                    )
                await disconnected_event.wait()

if __name__ == "__main__":
    asyncio.run(main())
