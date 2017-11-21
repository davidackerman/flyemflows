import re
import glob

import numpy as np

from PIL import Image

from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid
from DVIDSparkServices.dvid.metadata import DataInstance, get_blocksize

from .volume_service import VolumeServiceReader, VolumeServiceWriter


class SliceFilesVolumeServiceReader(VolumeServiceReader):

    def __init__(self, volume_config):
        self._preferred_message_shape_zyx = volume_config["geometry"]["message-block-shape"][::-1]
        assert self._preferred_message_shape_zyx[0] in (1,-1) and self._preferred_message_shape_zyx[1:] == [-1,-1], \
            "Preferred message shape for slice fils must be a single Z-slice, and a complete XY plane"

        slice_fmt = volume_config["slice-files"]["slice-path-format"]
        slice_xy_offset = volume_config["slice-files"]["slice-xy-offset"]

        if '%' in slice_template:
            raise RuntimeError("Please use python-style string formatting for 'basename': (e.g. zcorr.{:05d}.png)")
        match = re.match('^(.*)({[^}]*})(.*)$', options["basename"])
        if not match:
            raise RuntimeError(f"Unrecognized format string for image basename: {options['basename']}")

        prefix, _index_format, suffix = match.groups()
        matching_paths = sorted( glob.glob(f"{prefix}*{suffix}") )
        path_lengths = 

        minslice = int( matching_paths[0][len(prefix):-len(suffix)] )
        maxslice = int( matching_paths[-1][len(prefix):-len(suffix)] )

        all_slices = glob.glob(slice_fmt.format())

        self._bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        assert -1 not in self._bounding_box_zyx.flat[:], \
            "volume_config must specify explicit values for bounding-box"
        
        self._server = volume_config["dvid"]["server"]
        self._uuid = volume_config["dvid"]["uuid"]

        if "segmentation-name" in volume_config["dvid"]:
            self._instance_name = volume_config["dvid"]["segmentation-name"]
            self._dtype = np.uint64
        elif "grayscale-name" in volume_config["dvid"]:
            self._instance_name = volume_config["dvid"]["grayscale-name"]
            self._dtype = np.uint8
            
        self._dtype_nbytes = np.dtype(self._dtype).type().nbytes

        data_instance = DataInstance(self._server, self._uuid, self._instance_name)
        self._instance_type = data_instance.datatype
        self._is_labels = data_instance.is_labels()

        block_shape = get_blocksize(self._server, self._uuid, self._instance_name)
        self._block_width = block_shape[0]
        assert block_shape[0] == block_shape[1] == block_shape[2], \
            "Expected blocks to be cubes."

        config_block_width = volume_config["geometry"]["block-width"]
        assert config_block_width in (-1, self._block_width), \
            f"DVID volume block-width ({config_block_width}) from config does not match server metadata ({self._block_width})"
        
        # Overwrite config values
        volume_config["geometry"]["block-width"] = self._block_width

    @property
    def dtype(self):
        return self._dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def block_width(self):
        return self._block_width
    
    @property
    def bounding_box_zyx(self):
        return self._bounding_box_zyx

    def get_subvolume(self, box_zyx, scale=0):
        shape = np.asarray(box_zyx[1]) - box_zyx[0]
        req_bytes = self._dtype_nbytes * np.prod(box_zyx[1] - box_zyx[0])
        throttle = (self._resource_manager_client.server_ip == "")
        with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
            return sparkdvid.get_voxels( self._server, self._uuid, self._instance_name,
                                         scale, self._instance_type, self._is_labels,
                                         shape, box_zyx[0],
                                         throttle=throttle )

class DvidVolumeServiceWriter(DvidVolumeServiceReader, VolumeServiceWriter):
    
    # Two-levels of auto-retry:
    # 1. Auto-retry up to three time for any reason.
    # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in ex.args[0] or '504' in ex.args[0])
    @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
    def write_subvolume(self, subvolume, offset_zyx, scale):
        req_bytes = self._dtype_nbytes * np.prod(subvolume.shape)
        throttle = (self._resource_manager_client.server_ip == "")
        with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
            return sparkdvid.post_voxels( self._server, self._uuid, self._instance_name,
                                          scale, self._instance_type, self._is_labels,
                                          subvolume, offset_zyx,
                                          throttle=throttle )
