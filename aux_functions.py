from astropy.io import fits
import numpy as np
from radio_beam import Beam

def compute_physical_resolution(pix_size, distance, verbose=False):
    """Compute the physical resolution of an image at a given distance

    Args:
        pix_size (float): Pixel size of the image [deg]
        distance (float): Distance of the object

    Returns:
        float: physical resolution
    """
    phys_res = 2 * distance * np.tan(np.radians(pix_size/2))
    if verbose:
        print(f"Pixel Dimension [deg]: {pix_size}")
        print(f"Distance to the galaxy: {distance}")
        print(f"Physical Resolution of the image at the distance of the galaxy [same units as distance]: {phys_res}")
    return phys_res
def create_ellipse(rx=15, ry=15,rz=1, res_points=100):
    # place an ellipse in the middle of the  plot to indicate the position of the galaxy disk
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, res_points)
    v = np.linspace(0, np.pi, res_points)
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    ell_x = rx * np.outer(np.cos(u), np.sin(v))
    ell_y = ry * np.outer(np.sin(u), np.sin(v))
    ell_z = rz * np.outer(np.ones_like(u), np.cos(v))
    return ell_x, ell_y, ell_z

def load_fits_map(file_path, hdu=0):
    """Loads the fits file containing the polarization angle data.

    Args:
        file_path (_type_): Path to fits file.
        hdu (int, optional): HDU extension where the data is stored. Defaults to 0.

    Returns:
        1. Flattened data array,
        2. Image dimensions,
        3. Radio Beam object,
        4. Pixel dimensions [deg, deg] that were stored in the fits-Header (CDELT1/2)
        5. fits Header
    """
    fits_hdul = fits.open(file_path)
    fits_hdul.info()
    data = fits_hdul[hdu].data
    data = np.squeeze(data)
    header = fits_hdul[hdu].header
    naxis1 = header['NAXIS1']
    naxis2 = header['NAXIS2']
    img_dimensions = [naxis1,naxis2]
    img_beam = Beam.from_fits_header(header)
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']
    pix_dim = [cdelt1, cdelt2]
    
        # Summarize all returns
    summary = (
        f"------------\n"
        f"Load Fits Map summary:\n"
        f"Flattened data array shape: {data.shape},\n"
        f"Image dimensions: {img_dimensions},\n"
        f"Radio Beam object: {img_beam},\n"
        f"Pixel dimensions [deg, deg]: {pix_dim},\n"
        f"Fits Header keys: {list(header.keys())[:10]} (first 10 keys)\n"
        f"------------\n")
    print(summary)
    
    return data, img_dimensions, img_beam, pix_dim, header

def scale_array(input_array, x):
    """
    Scales down a 2D array by averaging values within each x by x block.

    Parameters:
        input_array (numpy.ndarray): The 2D array to be scaled down.
        x (int): The scale factor. The array will be reduced by averaging x by x blocks.

    Returns:
        numpy.ndarray: The scaled-down 2D array.

    Example:
        input_array = np.array([
            [1, 2, 3, ..., 450, 451],
            [1, 2, 3, ..., 450, 451],
            ...
            [1, 2, 3, ..., 450, 451]
        ])
        x = 10
        scaled_result = scale_array(input_array, x)
        # scaled_result will be an array of shape (46, 46)
    """
    scaled_array = []
    rows = input_array.shape[0]
    cols = input_array.shape[1]

    # Calculate the number of blocks in rows and cols
    num_blocks_rows = (rows + x - 1) // x
    num_blocks_cols = (cols + x - 1) // x

    # Iterate over the blocks
    for i in range(num_blocks_rows):
        scaled_row = []
        for j in range(num_blocks_cols):
            # Compute the average of the subarray with size x by x
            start_row = i * x
            end_row = min((i + 1) * x, rows)
            start_col = j * x
            end_col = min((j + 1) * x, cols)
            sub_array = input_array[start_row:end_row, start_col:end_col]
            scaled_value = np.nanmean(sub_array)
            scaled_row.append(scaled_value)
        scaled_array.append(scaled_row)

    return np.array(scaled_array)