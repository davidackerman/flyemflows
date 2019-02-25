import copy
import tempfile

import h5py
import numpy as np
import scipy.ndimage

import pytest
from skimage.util.shape import view_as_blocks

from neuclease.util import box_to_slicing

from flyemflows.volumes import VolumeService, Hdf5VolumeService, ScaledVolumeService


@pytest.fixture
def setup_hdf5_service():
    test_dir = tempfile.mkdtemp()
    test_file = f'{test_dir}/scaled-volume-test.h5'
    
    full_volume = np.random.randint(255, size=(256,256,256))
    with h5py.File(test_file, 'w') as f:
        f['volume'] = full_volume

    box_zyx = np.array([[0,0,0], [100,200,256]])
    box_xyz = box_zyx[:,::-1]

    RAW_VOLUME_DATA = full_volume[box_to_slicing(*box_zyx)]
    
    VOLUME_CONFIG = {
      "hdf5": {
        "path": test_file,
        "dataset": "volume"
      },
      "geometry": {
          "bounding-box": box_xyz.tolist(),
          "available-scales": [0] # Ensure only the first scale is used.
      }
    }
    
    # First, hdf5 alone
    h5_reader = Hdf5VolumeService(VOLUME_CONFIG)
    assert (h5_reader.bounding_box_zyx == box_zyx).all()
    full_from_h5 = h5_reader.get_subvolume(h5_reader.bounding_box_zyx)
    assert full_from_h5.shape == (*(box_zyx[1] - box_zyx[0]),)
    assert (full_from_h5 == RAW_VOLUME_DATA).all()

    return RAW_VOLUME_DATA, VOLUME_CONFIG, full_from_h5, h5_reader

def test_api(setup_hdf5_service):
    _raw_volume, _volume_config, _full_from_h5, h5_reader = setup_hdf5_service
    scaled_reader = ScaledVolumeService(h5_reader, 0)
    assert scaled_reader.base_service == h5_reader
    assert len(scaled_reader.service_chain) == 2
    assert scaled_reader.service_chain[0] == scaled_reader
    assert scaled_reader.service_chain[1] == h5_reader

def test_full_volume_no_scaling(setup_hdf5_service):
    _raw_volume, _volume_config, full_from_h5, h5_reader = setup_hdf5_service
    
    scaled_reader = ScaledVolumeService(h5_reader, 0)
    assert (scaled_reader.bounding_box_zyx == h5_reader.bounding_box_zyx).all()
    assert (scaled_reader.preferred_message_shape == h5_reader.preferred_message_shape).all()
    assert scaled_reader.block_width == h5_reader.block_width
    assert scaled_reader.dtype == h5_reader.dtype

    full_scaled = scaled_reader.get_subvolume(scaled_reader.bounding_box_zyx)
    assert (full_scaled == full_from_h5).all()
    assert full_scaled.flags.c_contiguous

def test_full_volume_downsample_1(setup_hdf5_service):
    _raw_volume, volume_config, full_from_h5, h5_reader = setup_hdf5_service

    # Scale 1
    scaled_config = copy.deepcopy(volume_config)
    scaled_config["rescale-level"] = 1
    scaled_reader = VolumeService.create_from_config(scaled_config)
    
    assert (scaled_reader.bounding_box_zyx == h5_reader.bounding_box_zyx // 2).all()
    assert (scaled_reader.preferred_message_shape == h5_reader.preferred_message_shape // 2).all()
    assert scaled_reader.block_width == h5_reader.block_width // 2
    assert scaled_reader.dtype == h5_reader.dtype

    full_scaled = scaled_reader.get_subvolume(scaled_reader.bounding_box_zyx)
    assert (full_scaled == scipy.ndimage.interpolation.zoom(full_from_h5, 0.5)).all()
    assert full_scaled.flags.c_contiguous
    
def test_full_volume_upsample_1(setup_hdf5_service):
    _raw_volume, _volume_config, full_from_h5, h5_reader = setup_hdf5_service

    # Scale -1
    scaled_reader = ScaledVolumeService(h5_reader, -1)
    assert (scaled_reader.bounding_box_zyx == h5_reader.bounding_box_zyx * 2).all()
    assert (scaled_reader.preferred_message_shape == h5_reader.preferred_message_shape * 2).all()
    assert scaled_reader.block_width == h5_reader.block_width * 2
    assert scaled_reader.dtype == h5_reader.dtype

    full_scaled = scaled_reader.get_subvolume(scaled_reader.bounding_box_zyx)
    assert (full_from_h5 == full_scaled[::2,::2,::2]).all()
    assert full_scaled.flags.c_contiguous

def test_subvolume_no_scaling(setup_hdf5_service):
    _raw_volume, _volume_config, full_from_h5, h5_reader = setup_hdf5_service
    
    box = np.array([[13, 15, 20], [100, 101, 91]])
    subvol_from_h5 = full_from_h5[box_to_slicing(*box)].copy('C')
    
    scaled_reader = ScaledVolumeService(h5_reader, 0)
    subvol_scaled = scaled_reader.get_subvolume(box)

    assert (subvol_scaled.shape == box[1] - box[0]).all()
    assert subvol_from_h5.shape == subvol_scaled.shape, \
        f"{subvol_scaled.shape} != {subvol_from_h5.shape}"
    assert (subvol_scaled == subvol_from_h5).all()
    assert subvol_scaled.flags.c_contiguous

def test_subvolume_downsample_1(setup_hdf5_service):
    _raw_volume, _volume_config, full_from_h5, h5_reader = setup_hdf5_service
    
    down_box = np.array([[13, 15, 20], [20, 40, 41]])
    up_box = 2*down_box
    up_subvol_from_h5 = full_from_h5[box_to_slicing(*up_box)]
    down_subvol_from_h5 = scipy.ndimage.interpolation.zoom(up_subvol_from_h5, 0.5)
    
    # Scale 1
    scaled_reader = ScaledVolumeService(h5_reader, 1)
    subvol_scaled = scaled_reader.get_subvolume(down_box)

    assert (subvol_scaled.shape == down_box[1] - down_box[0]).all()
    assert down_subvol_from_h5.shape == subvol_scaled.shape, \
        f"{subvol_scaled.shape} != {down_subvol_from_h5.shape}"
    assert (subvol_scaled == down_subvol_from_h5).all()
    assert subvol_scaled.flags.c_contiguous

def test_subvolume_upsample_1(setup_hdf5_service):
    _raw_volume, _volume_config, full_from_h5, h5_reader = setup_hdf5_service

    up_box = np.array([[13, 15, 20], [100, 101, 91]])
    full_upsampled_vol = np.empty( 2*np.array(full_from_h5.shape), dtype=h5_reader.dtype )
    up_view = view_as_blocks(full_upsampled_vol, (2,2,2))
    up_view[:] = full_from_h5[:, :, :, None, None, None]
    up_subvol_from_h5 = full_upsampled_vol[box_to_slicing(*up_box)]
            
    # Scale -1
    scaled_reader = ScaledVolumeService(h5_reader, -1)
    subvol_scaled = scaled_reader.get_subvolume(up_box)

    assert (subvol_scaled.shape == up_box[1] - up_box[0]).all()
    assert up_subvol_from_h5.shape == subvol_scaled.shape, \
        f"{subvol_scaled.shape} != {up_subvol_from_h5.shape}"
    assert (subvol_scaled == up_subvol_from_h5).all()
    assert subvol_scaled.flags.c_contiguous


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'tests.volumes.test_scaled_volume_service'])
