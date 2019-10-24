import nilearn as nil
import numpy as np
import nibabel as nib
from . import utilities

import warnings

from tqdm import tqdm


from nilearn import masking

class SearchLight:
    """ An object that represents a (searchlight) directional variability analysis
        for one set of data, typically a single condition of a single participant.

        Args:
            radius (int): radius (in voxels) for the searchlight
            data (np.array): a 4d array of data OR None
            func (function): the function that should be called for each sphere. Schurger's dva is used if None.
            mask: a 3d array OR a 3d nifti OR None. A brain mask is automatically computed if None.
    """
    def __init__(self, data = None, radius = 2, unit="vox", mask = "background", func = None):
        
        # init variables
        self.nifti = None
        self.return_data = None

        # set radius
        self.set_radius(radius)
        # set data
        self.set_data(data)
        # set function
        self.set_func(func)
        # set mask
        self.set_mask(mask)
 

    def set_radius(self, radius, unit="vox"):
        self.radius = radius

    def set_data(self, images):
        try:
            self.data = nil.image.concat_imgs(images).get_data()
            self.nifti = nil.image.concat_imgs(images)
        except:
            try:
                self.data = images.get_data()
                self.nifti = images
            except:
                self.data = data

    def set_mask(self, mask):
        if mask == None or mask == "background":
            try:
                self.mask = masking.compute_background_mask(self.nifti).get_data()
            except:
                self.mask = mask
                warnings.warn("Unable to generate mask using selected method. This is expected behavior if you initialze a SearchLight object without data. A mask using the selected method will be created when you call run(). If this message appears when calling run(), please check your data.")
        elif hasattr(mask, "get_data"):
            self.mask = mask.get_data()
        else:
            self.mask = mask
            

    def set_func(self, func):
        self.func = func

        

    def run(self):
        data = self.data

        # check if a mask has been set
        if isinstance(self.mask, str):
            self.set_mask(self.mask)

        try:
            # 'return_array' is an array matching the shape of of he first thre diemsnions of 'data'
            # for every voxel in this array
            return_array = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
            
            masker = SphericalMasker(radius = self.radius)

            voxelcount = np.count_nonzero(self.mask)

            print("INFO: {vox} Voxels in mask.".format(vox=voxelcount))

            with tqdm(total=voxelcount) as pbar:
                i = 0
                for (x, y, z), voxel in np.ndenumerate(return_array):
                    if self.mask[(x, y, z)] == 1:
                        # construct sphere around voxel and get voxel values
                        voxels_over_time = masker.extract_values(data, (x, y, z))

                        if voxels_over_time != -1:
                            voxels_matrix = np.stack(voxels_over_time, axis=0)
                            return_array[x, y, z] = self.func(voxels_matrix)
                        else:
                            return_array[x, y, z] = -1
                        
                        i = i +1
                        if i % 100 == 0: pbar.update(100)


            self.return_data = return_array
        except Exception as e: 
            print(e)
            print('No data to run searchlight on.')


    def get_return_nifti(self):
        if self.nifti == None:
            warnings.warn("Can only output as Nifti if a Nifti was supplied in the first place.", DeprecationWarning)
            return
        #TODO: check if results ready
        return nib.Nifti1Image(self.return_data, self.nifti.affine, nib.Nifti1Header())

    
class SphericalMasker:
    """ A masker to extract spheres from 4-diemsnioanl arrays """
    
    def __init__(self, radius = 2):
        """Constructs a 3-dimensional sphere with center m and the provided radius
            
            Args:
                radius (int): radius in voxels
        """
        import numpy as np
        # creates a mesh grid containing distances to m
        y, x, z = np.ogrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
        # make a binary mask (a.k.a. the sphere)
        self.mask = x ** 2 + y ** 2 + z ** 2 <= radius ** 2
        self.radius = radius

    def extract_values(self, data, m):
        """ Applies 'mask' to the 4d data array given and returns a list of arrays,
        each array representing the values of a voxel over the 4th dimension.

        Args:
            data (np.array): 4d array
            m (touple): 3d point, center of the sphere

        Returns:
            data_in_sphere (list): a list containing the values in the sphere over the 4th dimension
        """
        import numpy as np

        mask = self.mask
        radius = self.radius

        # get the values of all voxels in the sphere over the 4th dimension (beware of out_of_bounds!)
        try:
            return [
                data[m[0] - radius + x, m[1] - radius + y, m[2] - radius + z, :]
                for (x, y, z), e in np.ndenumerate(mask)
                if e
            ]
        except:
            return -1