"""
This script processes pixel data from the first pixel range and computes various metrics, saving the results into 48 separate files.
"""
import numpy as np
import healpy as hp
from tqdm import tqdm
import multiprocessing as mp
import math
from multiprocessing import Pool,Array,Lock
import time

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


#  Angle between two vector

def angle_vec(A, B):
    dot_product = np.dot(A, B)
    mag_A = np.linalg.norm(A)
    mag_B = np.linalg.norm(B)
    if (mag_A * mag_B) == 0:
        return 0 # To handle the case where one the vector becomes Zeros(R_i == Rc)
    cos_theta = dot_product / (mag_A * mag_B)
    angle = np.arccos(cos_theta)
    return angle

def anglev(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    clipped_dp = np.clip(dot_product, -1.0, 1.0) # Clip dot_product to the valid range for arccos to avoid NaNs
    angle = np.arccos(clipped_dp)
    return angle


nside = 1024
npix = 12*nside**2

# time_step=scan_time
scan_time = np.sqrt(4*np.pi/npix)/w1

# temperature_map = hp.read_map("input_map.fits")

# Load the grid
grid = np.loadtxt("grid.txt")
# grid_size = 2 (in arcsec)convert grid_size in radian
grid_size = 2*math.pi / (180 * 3600)
centre = (3001,3001)
Radius = (50 / 60) * (math.pi / 180) #50 in arcmin

def process_time_step(time_step,R,S,pix_ring):

    #3. Calculate Z, I and N (N = I for phi = 0)
    Z_t = np.cross(R,S)
    I_t = np.cross(R, Z_t)
    N_t = I_t

    # 4. Find neighboring pixels in RING format
    Rc = hp.pix2vec(nside,pix_ring,nest=False)
    neighbours = hp.query_disc(nside, Rc , radius=Radius)

    # 5. angular separation between central pixel and neighbouring pixels
    x = np.zeros_like(neighbours, dtype=float)
    y = np.zeros(len(neighbours))
    weight = np.zeros(len(neighbours))
    # print(len(neighbours))
    for i, neighbour_pix in enumerate(neighbours):

        R_i = hp.pix2vec(nside,neighbour_pix,nest=False)
        theta_i = anglev(Rc, R_i)

        # 6. A_i = line joining central pixel and neighbour pixel
        R_i = hp.pix2vec(nside,neighbour_pix,nest=False)
        A_i = np.array(Rc)-np.array(R_i)
        # print("A_i = ",A_i,"\nN_t = ",N_t)

        # 7. angle between N & A_i
        alpha_i = angle_vec(A_i, N_t)
        # print("alpha_i=",(alpha_i))
        # 8. x_i and y_i
        x[i] = theta_i * np.cos(alpha_i)
        y[i] = theta_i * np.sin(alpha_i)
        index_x = int(centre[0] + round(x[i]/grid_size,0))
        index_y = int(centre[1] + round(y[i]/grid_size,0))
        weight[i] = grid[index_x][index_y]
        # print(centre[0] )
    dictionary = {pix: weight[i] for i, pix in enumerate(neighbours)}
    result = np.array(list(dictionary.items())).flatten()

    return pix_ring, result
    # return {pix: weight[i] for i, pix in enumerate(neighbours)}

start = 0
duration = 60*24*365 # in min (one month)
steps = int(duration / scan_time)
# steps = 10000
length = 700 # no. of neigbours


# time_periods = np.linspace(start, start + duration, steps)
# time_periods_iterator = tqdm(time_periods, desc="Processing", total=len(time_periods))

# To make an array having  occurance, centre_pix, neighbors&weights (size = 2 + no. of neighbour*2)
import numpy as np
format_string = '\t'.join(['%d', '%.8e'] * length)
centre_pix_format = "%d"
occurance_format = "%d"


fmt = [occurance_format] + [centre_pix_format] + format_string.split('\t')  # Split format_string by tabs

def process_chunk(chunk_range, start1,duration1,steps1):
    local_array_map = np.zeros((chunk_range[1] - chunk_range[0], (2 * length + 2)))
    local_array_map[:, 1] = np.arange(chunk_range[0], chunk_range[1])
    local_array_map[:, 0] = 0
    local_array_map[:, 2] = 0

    time_periods = np.linspace(start1, start1 + duration1, steps1)

    start_time_chunk = time.time()
    for time_period in tqdm(time_periods, desc=f"Processing chunk {chunk_range[0]} to {chunk_range[1]}"):
        R, S = get_vectors(time_period)
        pixel = hp.vec2pix(nside, R[0], R[1], R[2], nest=False)
        if chunk_range[0] <= pixel < chunk_range[1]:
            idx = pixel - chunk_range[0]
            pixel, weight = process_time_step(time_period, R, S, pixel)
            result = np.pad(weight, (0, 2 * length - len(weight)), mode='constant', constant_values=0)
            local_array_map[idx, 2:] += result
            local_array_map[idx, 0] += 1  # occurrence = occurrence + 1

    end_time_chunk = time.time()
    elapsed_time_chunk = end_time_chunk - start_time_chunk
    print(f"Chunk {chunk_range[0]} to {chunk_range[1]} took {elapsed_time_chunk:.2f} seconds")

    # Save local_array_map to a separate file with naming convention based on chunk range
    filename = f"1new_1024_{chunk_range[0]}_{chunk_range[1]}.dat"
    np.savetxt(filename, local_array_map, fmt=fmt)

    return None  # Since we're saving data directly, no need to return local_array_map


def main():
    pixel_ranges = [
    3735, 2831, 3532, 4391, 5013, 5401, 5840, 6017, 6396, 6448,
    6563, 6826, 6886, 6909, 6862, 7025, 6923, 6821, 6885, 6731,
    6475, 6511, 6125, 5728, 5098, 4434, 5646, 6226, 6818, 7114,
    7416, 7872, 8174, 8414, 8950, 8942, 9435, 9655, 9946, 10227,
    10421, 10931, 10978, 11333, 11564, 11938, 11981, 12408
    ]

    chunk_ranges = []
    current_start = 0
    for pixel_range in pixel_ranges:
        current_end = current_start + pixel_range
        chunk_ranges.append((current_start, current_end))
        current_start = current_end

    start_time = time.time()

    with mp.Pool(processes=48) as pool:
        pool.starmap(process_chunk, [(chunk_range, start,duration,steps) for chunk_range in chunk_ranges])

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Elapsed time2:", elapsed_time, "seconds")

if __name__ == "__main__":
    main()

