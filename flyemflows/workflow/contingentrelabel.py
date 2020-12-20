import copy
import pickle
import logging

import numpy as np
import pandas as pd
import dask.bag as db

from dvid_resource_manager.client import ResourceManagerClient
from neuclease.util import Timer, round_box, SparseBlockMask, boxes_from_grid, iter_batches, extract_subvol, box_intersection
from neuclease.dvid import fetch_roi

from ..util import replace_default_entries
from ..volumes import VolumeService, VolumeServiceWriter, SegmentationVolumeSchema, DvidVolumeService, ScaledVolumeService
from .util.config_helpers import BodyListSchema, load_body_list
from . import Workflow

logger = logging.getLogger(__name__)


class ContingentRelabel(Workflow):
    """
    Relabel a volume, contingent on the labels in a different volume.

    Given two segmentation volumes ("primary-input" and "contingent-input")
    and a mapping from (primary, contingent) -> final-primary,
    relabel the 'primary' volume according to the mapping,
    using both the 'primary' and 'contingent' labels at each voxel
    position as the mapping key.

    To produce a suitable contingent-mapping for this workflow from a contingency table, see flyemflows/workflow/contingent_mapping.py
    """
    ContingentRelabelOptionsSchema = {
        "type": "object",
        "description": "Settings specific to the ContingentRelabel workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "roi": {
                "description": "Limit analysis to bricks that intersect the given ROI, \n"
                               "which must come from the same DVID node as the primary-input source.\n"
                               "(Only valid when the primary-input is a DVID source.)",
                "type": "string",
                "default": ""
            },
            "subset-labels": {
                **BodyListSchema,
                "description": "If provided, only the listed labels will be analyzed.\n"
                               "Other labels will be left untouched in the results.\n",
            },
            "skip-sparse-fetch": {
                "description": "If True, do not attempt to fetch the sparsevol-coarse for the given subset-labels.\n"
                               "Just fetch the entire bounding-box.\n",
                "type": "boolean",
                "default": False
            },
            "batch-size": {
                "description": "Bricks will be relabeled in batches.\n"
                               "This setting specifies the number of bricks in each batch.\n",
                "type": "integer",
                "default": 1000
            },
            "write-changed-bricks-only": {
                "description": "Write only bricks that have voxels that were relabeled; leave others untouched.\n"
                               "(Useful optimization if your input and output are identical at first.)\n",
                "type": "boolean",
                "default": False
            },
            "contingent-mapping": {
                "description": "Path to a csv or npy file containing the mapping with columns [primary,contingency,final]\n",
                "type": "string"
                # NO DEFAULT
            },
            "mapping-broadcast": {
                "description": "How to distribute the mapping to the workers.\n",
                "type": "string",
                "enum": ["network", "nfs"],
                "default": "nfs"
            }
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "primary-input": SegmentationVolumeSchema,
        "contingency-input": SegmentationVolumeSchema,
        "output": SegmentationVolumeSchema,
        "contingentrelabel": ContingentRelabelOptionsSchema
    })

    @classmethod
    def schema(cls):
        return ContingentRelabel.Schema

    def execute(self):
        self.init_services()

        primary_service = self.primary_service
        contingency_service = self.contingency_service
        output_service = self.output_service
        options = self.config["contingentrelabel"]

        primary_is_supervoxels = False
        if isinstance(primary_service.base_service, DvidVolumeService):
            primary_is_supervoxels = primary_service.base_service.supervoxels

        roi = options["roi"]
        subset_labels = load_body_list(options["subset-labels"], primary_is_supervoxels)
        subset_labels = set(subset_labels)
        sparse_fetch = not options["skip-sparse-fetch"]

        # Boxes are determined by the primary volume/labels/roi
        boxes = self.init_boxes( primary_service,
                                 sparse_fetch and subset_labels,
                                 roi )

        batches = iter_batches(boxes, options["batch-size"])
        logger.info(f"Relabeling {len(boxes)} bricks in total.")
        logger.info(f"Processing {len(batches)} batches of {options['batch-size']} bricks each.")

        def _contingent_relabel(box):
            primary_vol = primary_service.get_subvolume(box)
            primary_vol = np.ascontiguousarray(primary_vol)

            contingency_vol = contingency_service.get_subvolume(box)
            contingency_vol = np.ascontiguousarray(contingency_vol)

            # Get the set of labels in this box, so we can discard irrelevant portions of the mapping.
            _primary_labels = pd.unique(primary_vol.reshape(-1))  # noqa
            _contingency_labels = pd.unique(contingency_vol.reshape(-1))  # noqa

            cm_path = options["contingent-mapping"]
            if cm_path.endswith('.npy'):
                _cm = np.load(options["contingent-mapping"])
            elif cm_path.endswith('.pkl'):
                _cm = pickle.load(open(cm_path, 'rb'))
            else:
                raise RuntimeError(f"Don't know how to open mapping file: {cm_path}")

            cm_df = pd.DataFrame(_cm)
            assert {*cm_df.columns} == {'primary', 'contingency', 'final'}

            # Keep only the parts of the mapping we need for this box,
            # just for the sake of performance in the merge below.
            cm_df = cm_df.query('primary in @_primary_labels and contingency in @_contingency_labels').copy()
            cm_df['primary'] = cm_df['primary'].astype(primary_vol.dtype)
            cm_df['contingency'] = cm_df['contingency'].astype(contingency_vol.dtype)

            # Use a merge to essentially map from (primary, contingency) -> final
            input_df = pd.DataFrame({'primary': primary_vol.reshape(-1),
                                     'contingency': contingency_vol.reshape(-1)})
            input_df = input_df.merge(cm_df, 'left', on=['primary', 'contingency'])
            input_df['final'] = input_df['final'].fillna(input_df['primary'])
            input_df['final'] = input_df['final'].astype(primary_vol.dtype)

            final_vol = input_df['final'].values.reshape(primary_vol.shape)
            del input_df

            output_service.write_subvolume(final_vol, box[0])

        for batch_index, batch_boxes in enumerate(batches):
            with Timer(f"Batch {batch_index}: Relabeling", logger):
                # Aim for 4 partitions per worker
                total_cores = sum( self.client.ncores().values() )
                (db.from_sequence(batch_boxes, npartitions=4*total_cores)
                   .map(_contingent_relabel)
                   .compute())

    def init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        """
        primary_config = self.config["primary-input"]
        contingency_config = self.config["contingency-input"]
        output_config = self.config["output"]

        mgr_config = self.config["resource-manager"]

        self.resource_mgr_client = ResourceManagerClient( mgr_config["server"], mgr_config["port"] )
        self.primary_service = VolumeService.create_from_config( primary_config, self.resource_mgr_client )
        self.contingency_service = VolumeService.create_from_config( contingency_config, self.resource_mgr_client )

        # Replace 'auto' dimensions with input bounding box
        replace_default_entries(output_config["geometry"]["bounding-box"], self.primary_service.bounding_box_zyx[:, ::-1])
        replace_default_entries(output_config["geometry"]["message-block-shape"], self.primary_service.preferred_message_shape[::-1])

        self.output_service = VolumeService.create_from_config( output_config, self.resource_mgr_client )

        assert isinstance(self.output_service, VolumeServiceWriter)

        if (self.primary_service.bounding_box_zyx != self.contingency_service.bounding_box_zyx).any():
            raise RuntimeError("Your primary and contingency input volumes do not have the same bounding box.  Please specify explicit bounding boxes.")

        if (self.output_service.bounding_box_zyx != self.primary_service.bounding_box_zyx).any():
            raise RuntimeError("Your output volume bounding box doesn't match the input volumes.  Please specify explicit bounding boxes.")

        logger.info(f"Bounding box: {self.primary_service.bounding_box_zyx[:,::-1].tolist()}")

        if (self.primary_service.preferred_message_shape != self.contingency_service.preferred_message_shape).any():
            raise RuntimeError("Your primary and contingency input volumes must use the same message-block-shape.")

        if (self.output_service.preferred_message_shape != self.primary_service.preferred_message_shape).any():
            raise RuntimeError("Your input and output volumes must use the same message-block-shape.")

    def init_boxes(self, volume_service, subset_labels, roi):
        sbm = None
        if roi:
            base_service = volume_service.base_service
            assert isinstance(base_service, DvidVolumeService), \
                "Can't specify an ROI unless you're using a dvid input"

            assert isinstance(volume_service, (ScaledVolumeService, DvidVolumeService)), \
                "The 'roi' option doesn't support adapters other than 'rescale-level'"
            scale = 0
            if isinstance(volume_service, ScaledVolumeService):
                scale = volume_service.scale_delta
                assert scale <= 5, \
                    "The 'roi' option doesn't support volumes downscaled beyond level 5"

            server, uuid, _seg_instance = base_service.instance_triple

            brick_shape = volume_service.preferred_message_shape
            assert not (brick_shape % 2**(5-scale)).any(), \
                "If using an ROI, select a brick shape that is divisible by 32"

            seg_box = volume_service.bounding_box_zyx
            seg_box = round_box(seg_box, brick_shape)
            seg_box_s5 = seg_box // 2**(5-scale)

            with Timer(f"Fetching mask for ROI '{roi}'", logger):
                roi_mask_s5, roi_box_s5 = fetch_roi(server, uuid, roi, format='mask')

            # Restrict to input bounding box
            clipped_roi_box_s5 = box_intersection(seg_box_s5, roi_box_s5)
            clipped_roi_mask_s5 = extract_subvol(roi_mask_s5, clipped_roi_box_s5-roi_box_s5[0])

            # Align to brick grid
            aligned_roi_box_s5 = round_box(clipped_roi_box_s5, brick_shape // 2**5, 'out')
            padding = (aligned_roi_box_s5 - clipped_roi_box_s5)
            padding[0] *= -1
            aligned_roi_mask_s5 = np.pad(clipped_roi_mask_s5, padding.transpose())

            # At the service native scale
            aligned_roi_box = (2**(5-scale)*aligned_roi_box_s5)
            logger.info(f"Brick-aligned ROI '{roi}' has bounding-box {aligned_roi_box[:, ::-1].tolist()}")

            # SBM 'full-res' corresponds to the input service voxels, not necessarily scale-0.
            sbm = SparseBlockMask.create_from_highres_mask( aligned_roi_mask_s5,
                                                            2**(5-scale),
                                                            aligned_roi_box,
                                                            brick_shape )
        elif subset_labels:
            try:
                sbm = volume_service.sparse_block_mask_for_labels([*subset_labels])
                if ((sbm.box[1] - sbm.box[0]) == 0).any():
                    raise RuntimeError("Could not find sparse masks for any of the subset-labels")
            except NotImplementedError:
                sbm = None

        if sbm is None:
            boxes = boxes_from_grid(volume_service.bounding_box_zyx,
                                    volume_service.preferred_message_shape,
                                    clipped=True)
            return np.array([*boxes])
        else:
            boxes = sbm.sparse_boxes(brick_shape)
            boxes = np.array(boxes)

            # Clip
            boxes[:,0,:] = np.maximum(volume_service.bounding_box_zyx[0], boxes[:,0,:])
            boxes[:,1,:] = np.minimum(volume_service.bounding_box_zyx[1], boxes[:,1,:])
            assert (boxes[:,0,:] < boxes[:,1,:]).all(), \
                "After cropping to input volume, some bricks disappeared."

            return boxes
