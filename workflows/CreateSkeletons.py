import copy
import json
import datetime

from DVIDSparkServices.util import bb_to_slicing, bb_as_tuple
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
from DVIDSparkServices.skeletonize_array import SkeletonConfigSchema, skeletonize_array
from DVIDSparkServices.reconutils.downsample import downsample_binary_3d, downsample_box

class CreateSkeletons(DVIDWorkflow):
    DvidInfoSchema = \
    {
      "type": "object",
      "properties": {
        "dvid-server": {
          "description": "location of DVID server",
          "type": "string",
          "minLength": 1,
          "property": "dvid-server"
        },
        "uuid": {
          "description": "version node to retrieve the segmentation from",
          "type": "string",
          "minLength": 1
        },
        "roi": {
          "description": "region of interest to skeletonize",
          "type": "string",
          "minLength": 1
        },
        "segmentation": {
          "description": "location of segmentation",
          "type": "string",
          "minLength": 1
        }
      },
      "required": ["dvid-server", "uuid", "roi", "segmentation"],
      "additionalProperties": False
    }
    
    SkeletonWorkflowOptionsSchema = copy.copy(Workflow.OptionsSchema)
    SkeletonWorkflowOptionsSchema["properties"].update(
    {
      "chunk-size": {
        "description": "Size of blocks to process independently (and then stitched together).",
        "type": "integer",
        "default": 512
      },
      "minimum-segment-size": {
        "description": "Size of blocks to process independently (and then stitched together).",
        "type": "integer",
        "default": 1000
      },
      "downsample-factor": {
        "description": "Factor by which to downsample bodies before skeletonization. (-1 means 'choose automatically')",
        "type": "integer",
        "default": -1 # -1 means "auto", based on RAM.
      }
    })
    
    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to create skeletons from segmentation",
      "type": "object",
      "properties": {
        "dvid-info": DvidInfoSchema,
        "skeleton-config": SkeletonConfigSchema,
        "options" : SkeletonWorkflowOptionsSchema
      }
    }

    @staticmethod
    def dumpschema():
        return json.dumps(CreateSkeletons.Schema)

    def __init__(self, config_filename):
        # ?! set number of cpus per task to 2 (make dynamic?)
        super(CreateSkeletons, self).__init__(config_filename, CreateSkeletons.dumpschema(), "CreateSkeletons")

    def execute(self):
        from pyspark import StorageLevel

        config = self.config_data
        chunksize = config["options"]["chunk-size"]

        # grab ROI subvolumes and find neighbors
        distsubvolumes = self.sparkdvid_context.parallelize_roi( config["dvid-info"]["roi"],
                                                                 chunksize,
                                                                 0,
                                                                 True )
        distsubvolumes.persist(StorageLevel.MEMORY_AND_DISK_SER)

        # grab seg chunks: (sv_id, seg)
        seg_chunks = self.sparkdvid_context.map_labels64( distsubvolumes,
                                                          config["dvid-info"]["segmentation"],
                                                          0,
                                                          config["dvid-info"]["roi"])

        # (sv_id, seg) -> (seg), that is, drop sv_id
        seg_chunks = seg_chunks.values()

        # (sv, segmentation)
        sv_and_seg_chunks = distsubvolumes.values().zip(seg_chunks)
        distsubvolumes.unpersist()

        def body_masks(sv_and_seg):
            import numpy as np
            import vigra
            subvolume, segmentation = sv_and_seg
            z1, y1, x1, z2, y2, x2 = subvolume.box_with_border
            sv_start, sv_stop = (z1, y1, x1), (z2, y2, x2)
            
            segmentation = vigra.taggedView(segmentation, 'zyx')
            consecutive_seg = np.empty_like(segmentation, dtype=np.uint32)
            _, maxlabel, bodies_to_consecutive = vigra.analysis.relabelConsecutive(segmentation, out=consecutive_seg)
            consecutive_to_bodies = { v:k for k,v in bodies_to_consecutive.items() }
            del segmentation
            
            # We don't care what the 'image' parameter is, but we have to give something
            image = consecutive_seg.view(np.float32)
            acc = vigra.analysis.extractRegionFeatures(image, consecutive_seg, features=['Coord<Minimum >', 'Coord<Maximum >', 'Count'])

            body_ids_and_masks = []
            for label in xrange(1, maxlabel+1): # Skip 0
                count = acc['Count'][label]
                min_coord = acc['Coord<Minimum >'][label].astype(int)
                max_coord = acc['Coord<Maximum >'][label].astype(int)
                box_local = np.array((min_coord, 1+max_coord))
                
                mask = (consecutive_seg[bb_to_slicing(*box_local)] == label).view(np.uint8)
                compressed_mask = CompressedNumpyArray(mask)

                body_id = consecutive_to_bodies[label]
                box_global = box_local + sv_start

                # Only keep segments that are big enough OR touch the subvolume border.
                if count >= config["options"]["minimum-segment-size"] \
                or (box_global[0] == sv_start).any() \
                or (box_global[1] == sv_stop).any():
                    body_ids_and_masks.append( (body_id, (bb_as_tuple(box_global), compressed_mask)) )
            
            return body_ids_and_masks


        # (sv, seg) -> (body_id, (box, mask))
        body_ids_and_masks = sv_and_seg_chunks.flatMap( body_masks )


        def combine_masks( boxes_and_compressed_masks ):
            import numpy as np
            boxes, _compressed_masks = zip(*boxes_and_compressed_masks)
            boxes = np.asarray(boxes)
            assert boxes.shape == (len(boxes_and_compressed_masks), 2,3)
            
            combined_box = np.zeros((2,3), dtype=np.int64)
            combined_box[0] = boxes[:, 0, :].min(axis=0)
            combined_box[1] = boxes[:, 1, :].max(axis=0)
            
            downsample_factor = config["options"]["downsample-factor"]
            if downsample_factor < 1:
                # FIXME: Auto-choose downsample factor if necessary
                downsample_factor = 1

            block_shape = np.array((downsample_factor,)*3)
            combined_downsampled_box = downsample_box( combined_box, block_shape )
            combined_downsampled_box_shape = combined_downsampled_box[1] - combined_downsampled_box[0]

            combined_mask_downsampled = np.zeros( combined_downsampled_box_shape, dtype=np.uint8 )
            
            for (box_global, compressed_mask) in boxes_and_compressed_masks:
                box_global = np.array(box_global)
                mask = compressed_mask.deserialize()
                mask_downsampled, downsampled_box = downsample_binary_3d(mask, downsample_factor, box_global)
                downsampled_box[:] -= combined_downsampled_box[0]

                combined_mask_downsampled[ bb_to_slicing(*downsampled_box) ] |= mask_downsampled

            if combined_mask_downsampled.sum() * downsample_factor**3 < config["options"]["minimum-segment-size"]:
                # 'None' results will be filtered out. See below.
                combined_mask_downsampled = None

            return ( combined_box, combined_mask_downsampled, downsample_factor )


        def combine_and_skeletonize(boxes_and_compressed_masks):
            (combined_box_start, _combined_box_stop), combined_mask, downsample_factor = combine_masks( boxes_and_compressed_masks )
            if combined_mask is None:
                return None
            tree = skeletonize_array(combined_mask, config["skeleton-config"])
            tree.rescale(downsample_factor, downsample_factor, downsample_factor, True)
            tree.translate(*combined_box_start[::-1]) # Pass x,y,z, not z,y,x
            
            # Also show which downsample factor was actually chosen
            config_copy = copy.deepcopy(config)
            if config_copy["options"]["downsample-factor"] < 1:
                config_copy["options"]["(dynamic-downsample-factor)"] = downsample_factor
            
            config_comment = json.dumps(config_copy, sort_keys=True, indent=4, separators=(',', ': '))
            config_comment = "\n".join( "# " + line for line in config_comment.split("\n") )
            config_comment += "\n\n"
            
            swc_contents =  "# {:%Y-%m-%d %H:%M:%S}\n".format(datetime.datetime.now())
            swc_contents += "# Generated by the DVIDSparkServices CreateSkeletons workflow.\n"
            swc_contents += "# Workflow config:\n"
            swc_contents += "# \n"
            swc_contents += config_comment + tree.toString()
            return swc_contents

        
        # (body_id, (box, mask))
        #   --> (body_id, [(box, mask), (box, mask), (box, mask), ...])
        #     --> (body_id, swc_contents)
        grouped_body_ids_and_masks = body_ids_and_masks.groupByKey()
        body_ids_and_skeletons = grouped_body_ids_and_masks.mapValues(combine_and_skeletonize)

        def post_swc_to_dvid( body_id_and_swc_contents ):
            body_id, swc_contents = body_id_and_swc_contents
            if swc_contents is None:
                return
        
            node_service = retrieve_node_service(config["dvid-info"]["dvid-server"],
                                                 config["dvid-info"]["uuid"],
                                                 config["options"]["resource-server"],
                                                 config["options"]["resource-port"])

            node_service.create_keyvalue("skeletons")
            node_service.put("skeletons", "{}_swc".format(body_id), swc_contents)

        body_ids_and_skeletons.foreach(post_swc_to_dvid)
