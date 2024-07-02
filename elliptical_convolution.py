import time
import numpy as np
import healpy as hp
from tqdm import tqdm
import multiprocessing

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

nside=1024
npix = 12*nside**2

# time_step=scan_time
scan_time = np.sqrt(4*np.pi/npix)/w1
fwhm_x = np.radians(10/60)
fwhm_y = np.radians(15/60)

sigma_x = fwhm_x / np.sqrt(8 * np.log(2))
sigma_y = fwhm_y / np.sqrt(8 * np.log(2))
sigma = max(sigma_x,sigma_y)

temperature_map = hp.read_map("input_map.fits")

def process_time_step(time_step):

    t = time_step

    # 1. Calculate R(t) and S(t) vectors
    R, S =  get_vectors(t)


    # 2. Calculate pixel number along R(t) vector (ring format)
    pix_ring = hp.vec2pix(nside, R[0], R[1], R[2], nest=False)


    #3. Calculate Z, I and N (N = I for phi = 0)
    Z_t = np.cross(R,S)
    I_t = np.cross(R, Z_t)
    N_t = I_t

    # 4. Find neighboring pixels in RING format
    Rc = hp.pix2vec(nside,pix_ring,nest=False)
    neighbours = hp.query_disc(nside, Rc , radius=(3*sigma))

    # 5. angular separation between central pixel and neighbouring pixels
    x = np.zeros_like(neighbours, dtype=float)
    y = np.zeros(len(neighbours))

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
        # print(x[i],theta_i * np.cos(alpha_i),y[i])

    # 9. Retrieve temperatures of neighboring pixels
    neighbor_temperatures = temperature_map[neighbours]
    # 10. Apply elliptical convolution
    convolved_temperature = np.sum(neighbor_temperatures * np.exp(-x**2 / (2 * sigma_x**2) -y**2 / (2 * sigma_y**2))) / np.sum(np.exp(-x**2 / (2 * sigma_x**2) -y**2 / (2 * sigma_y**2)))

    return int(pix_ring),convolved_temperature

start = time.time()

start_time=0
duration = 24*60*30 #in min (one month)
steps = int(duration / scan_time)


time_periods = np.linspace(start_time, start_time + duration,steps)

def parallel_execution(chunk):
    results = []
    for time_period in tqdm(chunk, desc="Processing"):
        pixel,temperature = process_time_step(time_period)
        results.append((time_period, pixel, temperature))
    return results


start = time.time()

# Split the time_periods array into chunks for parallel processing
chunks = np.array_split(time_periods, 48)

# Using multiprocessing for parallel execution
with multiprocessing.Pool(processes=48) as pool:
    results = pool.map(parallel_execution, chunks)

print("result processing")
# Flatten the results list of lists
results = [item for sublist in results for item in sublist]
# file_path = 'check.dat'
file_path = 'Data/1ellip.dat'
np.savetxt(file_path, results, fmt='%.4f %d %.16f ')
print(f"Results saved to {file_path}")
end = time.time()
elapsed_time = end - start
print(f"Total execution time: {elapsed_time:.2f} seconds")
