# Searchlight Analyis of Directional Varibility (dvasl)

## Introduction

## Installation
**dvasl** can be installed directly from GitHub using pip: `pip install git+https://github.com/marcpabst/dvasl`. 

## Example

```python
# 1. Import dependencies:
from dvasl.searchlight import SearchLight
from dvasl import utilities
import nilearn as nil
import nibabel as nib

# 2. Load data:
cond_a = nib.load("data/sub001_condA.nii.gz")

# 3. Create a SearchLight object:
sl_a = SearchLight( data = cond_a, # data, supplied as a Nifti-like object (see nibabel)
                    func = utilities.schurger_dva, # dva function to apply
                    radius = 2, # size of radius
                    unit = "vox", # in voxels
                    mask = "background") # create a brain mask using nilearn

# 4. Let's run the analysis:
sl_a.run() 

# 5. You can access the results as nifti-like:
raw_dva_map_a = sl.get_return_nifti()
```

One way to proceed from here is to calculate dva maps for two different conditions and then calculate the differen:

```python
nif = nil.image.math_img("img1 - img2", img1 = raw_dva_map_a, img2 = raw_dva_map_b)
nif.to_filename("sub1_diff_a_b.nii.gz")
```

