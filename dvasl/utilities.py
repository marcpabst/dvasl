import numpy as np
import nilearn as nil
import pandas as pd
import nibabel as nib
import scipy.stats


def schurger_dva(vectors):
    """ Calculates directional variability for the provided data. """
    
    vectors_normalized = np.empty(vectors.shape)
    for i in range(vectors.shape[1]):
        vector_norm = np.linalg.norm(vectors[:, i])
        vector_normalized = vectors[:, i] / vector_norm
        vectors_normalized[:, i] = vector_normalized

    dva = 1 - np.linalg.norm(np.sum(vectors_normalized, axis=1) / vectors.shape[1])

    return dva


def mean_l2norm(vectors):
    """ Calculates the mean vector norms """

    vectors_norms = np.empty(vectors.shape[1])
    for i in range(vectors.shape[1]):
        vectors_norms[i] = np.linalg.norm(vectors[:, i])

    mean_norm = np.mean(vectors_norms)

    return mean_norm



def run_searchlight(data, mask, radius, fun):
    """This function will, when called, do the following for each voxel inside
    'mask':
    1. Construct a 3-dimensional sphere around the voxel
    2. use that spehre to extract the timecourse of all the voxels inside that sphere
    3. call 'func', passing an np.array of shape voxels*time
    4. write the return value into an array (matching the size the first three dimensions
    of 'data')
    5. move to the next voxel

    Args:
        data (np.array): 4d array
        mask (np.array): 3d array, a binary mask representing voxel to analyse
        radius (int): radius of the sphere in voxels
        fun (function): the function to call

    Returns:
        return_array (np.array): an array matching the size of the first three dimensions
        of 'data', containing the output of 'func' for each sphere / voxel.
    """
    import nibabel as nib
    import nilearn as nil
    import nistats as nis

    # 'return_array' an array matching the shape of of he first thre diemsnions of 'data'
    # for every voxel in this array
    return_array = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    

    count = 0
    for (x, y, z), voxel in np.ndenumerate(return_array):
        if mask[(x, y, z)] == 1:
            # construct sphere around voxel and get voxel values
            voxels_over_time = construct_sphere(data, (x, y, z), radius)

            if voxels_over_time != -1:
                voxels_matrix = np.stack(voxels_over_time, axis=0)
                return_array[x, y, z] = fun(voxels_matrix)
            else:
                return_array[x, y, z] = -1

    return return_array

def construct_sphere(data, m, radius):
    """Constructs a 3-dimensional sphere with the center m and the provided radius,
    then applies that sphere to the 4d data array given and returns a list of arrays,
    each array representing the values of a voxel over the 4th dimension.

    Args:
        data (np.array): 4d array
        m (touple): 3d point, center of the sphere
        radius (int): radius of thr sphere in voxels

    Returns:
        data_in_sphere (list): a list containing the values in the sphere over the 4th dimension
    """
    # creates a mesh grid containing distances to m
    y, x, z = np.ogrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    # make a binary mask (a.k.a. the sphere)
    mask = x ** 2 + y ** 2 + z ** 2 <= radius ** 2

    # get the values of all voxels in the sphere over the 4th dimension (beware of out_of_bounds!)
    try:
        return [
            data[m[0] - radius + x, m[1] - radius + y, m[2] - radius + z, :]
            for (x, y, z), e in np.ndenumerate(mask)
            if e
        ]
    except:
        return -1

def ttest3d(images1, images2, mask=None):
    """ Calculates a voxel-wise t-test. For basic usage, the results 
    are identical to AFNI's 3dttest++. 
    """
    from scipy.stats import ttest_rel
    import nilearn as nil
    import numpy as np

    if hasattr(images1, "__iter__") and hasattr(images2, "__iter__"):
        data_array1 = nil.image.concat_imgs(images1).get_data()
        data_array2 = nil.image.concat_imgs(images2).get_data()
    elif isinstance(images1, np.Array) and isinstance(images2, np.Array):
        data_array1 = images1
        data_array2 = images2
    else:
        raise TypeError(
            "images1 and images2 must be either iterables of Niimg-like OR numpy-arrays."
        )

    if not data_array1.shape == data_array2.shape:
        raise ValueError("Array shapes don't match.")

    result_map_t = np.zeros((data_array1.shape[0], data_array1.shape[1], data_array1.shape[2]))
    result_map_p = np.zeros((data_array1.shape[0], data_array1.shape[1], data_array1.shape[2]))

    for (x, y, z), _ in np.ndenumerate(data_array1[:, :, :, 1]):
        if (not mask) or (mask[(x, y, z)]):
            result_map_t[(x, y, z)], result_map_p[(x, y, z)] = ttest_rel(
                data_array1[x, y, z, :], data_array2[x, y, z, :]
            )

    # return (result_map_t, result_map_p)
    return (
        nib.Nifti1Image(result_map_t, images1[0].affine),
        nib.Nifti1Image(result_map_p, images1[0].affine),
    )


def ttest3d_with_covariate(images1, images2, cov1, cov2, mask=None):
    """ Calculates a voxel-wise t-test with a covariate.

    For basic usage, the results should be identical to
    AFNI's 3dttest++. Note that this function will calculate
    the differences for voxels (samples and covariates)
    and then perform what is essentially a one-sample t-test
    with an added covariate.
    """
    from scipy.stats import ttest_rel
    import nilearn as nil
    import numpy as np
    if (
        hasattr(images1, "__iter__")
        and hasattr(images2, "__iter__")
        and hasattr(cov1, "__iter__")
        and hasattr(cov2, "__iter__")
    ):
        data_array1 = nil.image.concat_imgs(images1).get_data()
        data_array2 = nil.image.concat_imgs(images2).get_data()
        cov_array1 = nil.image.concat_imgs(cov1).get_data()
        cov_array2 = nil.image.concat_imgs(cov2).get_data()
    elif (
        isinstance(images1, np.Array)
        and isinstance(images2, np.Array)
        and isinstance(cov1, np.Array)
        and isinstance(cov2, np.Array)
    ):
        data_array1 = images1
        data_array2 = images2
        cov_array1 = cov1
        cov_array2 = cov2
    else:
        raise TypeError(
            "images1 and images2, co1 and cov2 must be either iterables of Niimg-like or numpy-arrays."
        )

    if not data_array1.shape == data_array2.shape == cov_array1.shape == cov_array2.shape:
        raise ValueError("Array shapes don't match.")


    # Create arrays that will be returned later
    result_map_t = np.zeros((data_array1.shape[0], data_array1.shape[1], data_array1.shape[2]))
    result_map_p = np.zeros((data_array1.shape[0], data_array1.shape[1], data_array1.shape[2]))

    # Iterate over every voxel
    for (x, y, z), _ in np.ndenumerate(data_array1[:, :, :, 1]):
        # If a mask is provided, only iterate over voxels in that mask:
        if (not mask) or (mask[(x, y, z)]):
            import statsmodels.api as sm

            # This creates a linear model for each voxel:
            #
            # Y = β₀ + β1X₁
            #
            # with Y being the difference in the samples (images1 and images2)
            # and X₁ being the difference in the covariate (cov1 and cov2)
            # (if X₁ = 0, this is equivalent to performing a simple t-test).
            #
            # Oh, we're also centering the covariate, because that's what
            # AFNI's 3dttest++ does.
            cov_diff = cov_array1[x, y, z, :] - cov_array2[x, y, z, :]
            pred = sm.add_constant(cov_diff - np.mean(cov_diff), prepend=True)

            model = sm.OLS(data_array1[x, y, z, :] - data_array2[x, y, z, :], pred)

            result = model.fit()

            result_map_t[(x, y, z)] = result.tvalues[0]
            result_map_p[(x, y, z)] = result.pvalues[0]

    return (
        nib.Nifti1Image(result_map_t, images1[0].affine),
        nib.Nifti1Image(result_map_p, images1[0].affine),
    )