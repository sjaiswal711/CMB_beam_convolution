# This script calculates and stores the total number of pixel occurrences over a year, divided into 48 files. 
# Each file contains the pixel data in between a certain range and their total occurrences in a year.

import numpy as np
import healpy as hp
from tqdm import tqdm
import multiprocessing
import math

theta1 = 7.5 * np.pi / 180
theta2 = 85 * np.pi / 180
w1 = 2 * np.pi  # rad/min
w2 = 2 * w1  # rad/min
w3 = 0.000011954  # rad/min

def get_vectors(t):
    cos_w1t = np.cos(w1 * t)
    sin_w1t = np.sin(w1 * t)

    cos_w2t = np.cos(w2 * t)
    sin_w2t = np.sin(w2 * t)

    cos_theta1 = np.cos(theta1)
    sin_theta1 = np.sin(theta1)

    cos_theta2 = np.cos(theta2)
    sin_theta2 = np.sin(theta2)

    A = np.array([[cos_w1t, sin_w1t, 0],
                  [-sin_w1t, cos_w1t, 0],
                  [0, 0, 1]])

    B = np.array([[1, 0, 0],
                  [0, cos_w2t, sin_w2t],
                  [0, -sin_w2t, cos_w2t]])

    C = np.array([[cos_theta1, 0, sin_theta1],
                  [0, 1, 0],
                  [-sin_theta1, 0, cos_theta1]])

    D_R = np.array([[cos_theta2],
                    [sin_theta2 * np.cos(w3 * t)],
                    [sin_theta2 * np.sin(w3 * t)]])

    D_S = np.array([[1],
                    [0],
                    [0]])

    result1 = np.dot(np.dot(A, B), C)
    result_R = np.matmul(result1, D_R)
    result_S = np.matmul(result1, D_S)

    return result_R.T.flatten(), result_S.T.flatten()  # Return both flattened vectors

nside=1024
npix = 12*nside**2

# time_step=scan_time
scan_time = np.sqrt(4*np.pi/npix)/w1

start_time=0
duration = 60*24*365  # in min (one month)
steps = int(duration / scan_time)
occurance_format = "%d"
centre_pix_format = "%d"

fmt = [occurance_format] + [centre_pix_format]

time_periods = np.linspace(start_time, start_time + duration, steps)

def parallel_execution(chunk, chunk_index):
    local_array_map = np.zeros((npix, 2))
    local_array_map[:, 1] = np.arange(0,npix)
    local_array_map[:, 0] = 0


    for time_period in tqdm(chunk, desc="Processing"):
        R, S = get_vectors(time_period)
        pixel = hp.vec2pix(nside, R[0], R[1], R[2], nest=False)
        idx = pixel
        local_array_map[idx, 0] += 1

    filename = f"occurance_{chunk_index + 1}.dat"
    np.savetxt(filename, local_array_map, fmt=fmt)

# Split the time_periods array into chunks
chunks = np.array_split(time_periods, 48)


with multiprocessing.Pool(processes=48) as pool:
    pool.starmap(parallel_execution, [(chunk, i) for i, chunk in enumerate(chunks)])

print("Result processing complete")
