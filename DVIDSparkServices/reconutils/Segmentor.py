"""Defines base class for segmentation plugins."""

class Segmentor(object):
    """Contains functionality for segmenting large datasets.
    
    It implements a very crude watershed algorithm by default.
    Users are expected to overwrite the segmentation function
    at least.  The other functions involving stitching the
    subvolumes and performing other RDD and DVID manipulations.

    Plugins should be placed in DVIDSparkServices.reconutils.plugins.

    """

    # determines what pixels are used as seeds (0-255 8-bit range)
    SEED_THRES = 150 
   
    # the size of the seed needed
    SEED_SIZE_THRES = 5

    # background size (what to completely ignore)
    BACKGROUND_SIZE = 100

    def __init__(self, context, config, options):
        self.context = context

        mode = config["options"]["stitch-algorithm"]
        if mode == "none":
            self.stitch_mode = 0
        elif mode == "conservative":
            self.stitch_mode = 1
        elif mode == "medium":
            self.stitch_mode = 2
        elif mode == "aggressive":
            self.stitch_mode = 3
        else:
            raise Exception("Invalid stitch mode specified")

        # set default watershed options (add option to use distance transform)
        self.seed_threshold = 0
        self.seed_size = 5
        self.seed_channels = [0]
        if "seed-threshold" in options:
            self.seed_threshold = options["seed-threshold"]
        if "seed-size" in options:
            self.seed_size = options["seed-size"]
        if "seed-channels" in options:
            self.seed_channels = options["seed-channels"]

    def predict_voxels(self, gray_chunks): 
        """Create a dummy placeholder boundary channel from grayscale.

        Takes an RDD of grayscale numpy volumes and produces
        an RDD of predictions (z,y,x,ch) and a watershed mask.
        """
        from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
        from scipy.ndimage import label
        from numpy import bincount 
        background_size = self.BACKGROUND_SIZE
        
        def _grayscale_identity(gray_chunk):
            (subvolume, gray) = gray_chunk
            import numpy

            # mask out background (don't have to 0 out seeds since)
            mask_mask = numpy.zeros(gray.shape).astype(numpy.uint8)
            mask_mask[gray == 0] = 1
            mask_mask = label(mask_mask)[0]
            mask_sizes = bincount(mask_mask.ravel())
            mask_components = mask_sizes < background_size 
            small_mask_locations = mask_components[mask_mask]
            mask_mask[small_mask_locations] = 0
            mask = mask_mask == 0 

            # invert and convert to 0 to 1 range for watershed
            new_gray = (255.0-gray)/255.0
            
            # convert it to a 1 channel prediction format
            pred = numpy.expand_dims(new_gray, 3) 
            
            pred_compressed = CompressedNumpyArray(pred)
            mask_compressed = CompressedNumpyArray(mask)

            return (subvolume, pred_compressed, mask_compressed)
        
        # preserver partitioner
        return gray_chunks.mapValues(_grayscale_identity)

    def create_supervoxels(self, prediction_chunks):
        """Performs watershed based on voxel prediction.

        Takes an RDD of numpy volumes with multiple prediction
        channels and produces an RDD of label volumes.  A mask must
        be provided indicating which parts of the volume should
        have a supervoxels (true to keep, false to ignore).  Currently,
        this is a seeded watershed, an option to use the distance transform
        for the watershed is forthcoming.  There are 3 hidden options
        that can be specified:
        
        Options (specified in JSON):
            seed-threshold = add seeds where boundary prob is <= threshold
            seed-size = seeds must be >= seed size to keep
            seed-channels (list[int]) = array of channels to use for watershed

        Args:
            prediction_chunks (RDD) = (subvolume key, (subvolume, 
                compressed numpy predictions, compressed numpy mask))
        Returns:
            watershed+predictions (RDD) as (subvolume key, (subvolume, 
                (numpy compressed array, numpy compressed array)))
        """
        from DVIDSparkServices.reconutils.morpho import seeded_watershed
        from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
        seed_threshold = self.seed_threshold
        seed_size = self.seed_size
        seed_channels = self.seed_channels

        def _watershed(prediction_chunks):
            (subvolume, prediction_compressed, mask_compressed) = prediction_chunks
            prediction = prediction_compressed.deserialize()
            if mask_compressed is None:
                mask = None
            else:
                assert isinstance( mask, CompressedNumpyArray )
                mask = mask_compressed.deserialize()

            # grab boundary data
            boundary = prediction[:,:,:,seed_channels[0]] 
           
            for bch in range(1, len(seed_channels)):
                boundary += prediction[:,:,:,seed_channels[bch]]

            supervoxels = seeded_watershed(boundary, seed_threshold,
                seed_size, mask)

            max_id = supervoxels.max()
            supervoxels_compressed = CompressedNumpyArray(supervoxels)
            subvolume.set_max_id(max_id)
            
            return (subvolume, prediction_compressed, supervoxels_compressed)
            
        # preserver partitioner
        return prediction_chunks.mapValues(_watershed)

    def agglomerate_supervoxels(self, sp_chunks):
        """No-op placeholder agglomeration of supervoxels.

        Args:
            seg_chunks (RDD) = (subvolume key, (subvolume, numpy compressed array, 
                numpy compressed array))
        Returns:
            segmentation (RDD) = (subvolume key, (subvolume, numpy compressed array))
        """
        
        def _noop_agglom(sp_chunk):
            (subvolume, prediction_compressed, supervoxels_compressed) = sp_chunk
            return (subvolume, supervoxels_compressed)

        # preserver partitioner
        return sp_chunks.mapValues(_noop_agglom)
    
    def segment(self, gray_chunks, checkpoint_dir = "", enable_pred_rollback = False):
        """Top-level pipeline (can overwrite) -- gray RDD => label RDD.

        Defines a segmentation workflow consisting of voxel prediction,
        watershed, and agglomeration.  One can overwrite specific functions
        or the entire workflow as long as RDD input and output constraints
        are statisfied.  RDD transforms should preserve the partitioner -- 
        subvolume id is the key.

        Args:
            gray_chunks (RDD) = (subvolume key, (subvolume, numpy grayscale))
            checkpoint_dir (str) = enables checkpointing for boundary prediction if not empty
            enable_pred_rollback (boolean) = enables rollback of prediction if available
        Returns:
            segmentation (RDD) as (subvolume key, (subvolume, numpy compressed array))

        """
        
        # run voxel prediction (default: grayscale is boundary)
        pred_chunks = self.predict_voxels(gray_chunks)

        # retrieve previously computed RDD or save current RDD
        if checkpoint_dir != "":
            pred_chunks = self.context.checkpointRDD(pred_chunks, 
                    checkpoint_dir, enable_pred_rollback)  

        # run watershed from voxel prediction (default: seeded watershed)
        sp_chunks = self.create_supervoxels(pred_chunks)

        # run agglomeration (default: none)
        segmentation = self.agglomerate_supervoxels(sp_chunks) 

        return segmentation 


    # label volumes to label volumes remapped, preserves partitioner 
    def stitch(self, label_chunks):
        from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
        
        def example(stuff):
            return stuff[1][0]
        
        # return all subvolumes back to the driver
        # create offset map (substack id => offset) and broadcast
        subvolumes = label_chunks.map(example).collect()
        offsets = {}
        offset = 0
        for subvolume in subvolumes:
            offsets[subvolume.roi_id] = offset
            offset += subvolume.max_id
        subvolume_offsets = self.context.sc.broadcast(offsets)

        # (key, subvolume, label chunk)=> (new key, (subvolume, boundary))
        def extract_boundaries(key_labels):
            # compute overlap -- assume first point is less than second
            def intersects(pt1, pt2, pt1_2, pt2_2):
                if pt1 > pt2:
                    raise Exception("point 1 greater than point 2")
                if pt1_2 > pt2_2:
                    raise Exception("point 1 greater than point 2")

                val1 = max(pt1, pt1_2)
                val2 = min(pt2, pt2_2)
                size = val2-val1
                npt1 = val1 - pt1 
                npt1_2 = val1 - pt1_2

                return npt1, npt1+size, npt1_2, npt1_2+size

            import numpy

            oldkey, (subvolume, labelsc) = key_labels
            labels = labelsc.deserialize()

            boundary_array = []
            
            # iterate through all ROI partners
            for partner in subvolume.local_regions:
                key1 = subvolume.roi_id
                key2 = partner[0]
                roi2 = partner[1]
                if key2 < key1:
                    key1, key2 = key2, key1
                
                # create key for boundary pair
                newkey = (key1, key2)

                # crop volume to overlap
                offx1, offx2, offx1_2, offx2_2 = intersects(
                                subvolume.roi.x1-subvolume.border,
                                subvolume.roi.x2+subvolume.border,
                                roi2.x1-subvolume.border,
                                roi2.x2+subvolume.border
                            )
                offy1, offy2, offy1_2, offy2_2 = intersects(
                                subvolume.roi.y1-subvolume.border,
                                subvolume.roi.y2+subvolume.border,
                                roi2.y1-subvolume.border,
                                roi2.y2+subvolume.border
                            )
                offz1, offz2, offz1_2, offz2_2 = intersects(
                                subvolume.roi.z1-subvolume.border,
                                subvolume.roi.z2+subvolume.border,
                                roi2.z1-subvolume.border,
                                roi2.z2+subvolume.border
                            )
                            
                labels_cropped = numpy.copy(labels[offz1:offz2, offy1:offy2, offx1:offx2])

                labels_cropped_c = CompressedNumpyArray(labels_cropped)
                # add to flat map
                boundary_array.append((newkey, (subvolume, labels_cropped_c)))

            return boundary_array


        # return compressed boundaries (id1-id2, boundary)
        mapped_boundaries = label_chunks.flatMap(extract_boundaries) 

        # shuffle the hopefully smallish boundaries into their proper spot
        # groupby is not a big deal here since same keys will not be in the same partition
        grouped_boundaries = mapped_boundaries.groupByKey()

        stitch_mode = self.stitch_mode

        # mappings to one partition (larger/second id keeps orig labels)
        # (new key, list<2>(subvolume, boundary compressed)) =>
        # (key, (subvolume, mappings))
        def stitcher(key_boundary):
            import numpy
            key, (boundary_list) = key_boundary

            # should be only two values
            if len(boundary_list) != 2:
                raise Exception("Expects exactly two subvolumes per boundary")
            # extract iterables
            boundary_list_list = []
            for item1 in boundary_list:
                boundary_list_list.append(item1)

            # order subvolume regions (they should be the same shape)
            subvolume1, boundary1_c = boundary_list_list[0] 
            subvolume2, boundary2_c = boundary_list_list[1] 

            if subvolume1.roi_id > subvolume2.roi_id:
                subvolume1, subvolume2 = subvolume2, subvolume1
                boundary1_c, boundary2_c = boundary2_c, boundary1_c

            boundary1 = boundary1_c.deserialize()
            boundary2 = boundary2_c.deserialize()

            if boundary1.shape != boundary2.shape:
                raise Exception("Extracted boundaries are different shapes")
            
            # determine list of bodies in play
            z2, y2, x2 = boundary1.shape
            z1 = y1 = x1 = 0 

            # determine which interface there is touching between subvolumes 
            if subvolume1.touches(subvolume1.roi.x1, subvolume1.roi.x2,
                                subvolume2.roi.x1, subvolume2.roi.x2):
                x1 = x2/2 
                x2 = x1 + 1
            if subvolume1.touches(subvolume1.roi.y1, subvolume1.roi.y2,
                                subvolume2.roi.y1, subvolume2.roi.y2):
                y1 = y2/2 
                y2 = y1 + 1
            
            if subvolume1.touches(subvolume1.roi.z1, subvolume1.roi.z2,
                                subvolume2.roi.z1, subvolume2.roi.z2):
                z1 = z2/2 
                z2 = z1 + 1

            eligible_bodies = set(numpy.unique(boundary2[z1:z2, y1:y2, x1:x2]))
            body2body = {}

            label2_bodies = numpy.unique(boundary2)

            # 0 is off,
            # 1 is very conservative (high percentages and no bridging),
            # 2 is less conservative (no bridging),
            # 3 is the most liberal (some bridging allowed if overlap
            # greater than X and overlap threshold)
            hard_lb = 50
            liberal_lb = 1000
            conservative_overlap = 0.90

            if stitch_mode > 0:
                for body in label2_bodies:
                    if body == 0:
                        continue
                    body2body[body] = {}

                # traverse volume to find maximum overlap
                for (z,y,x), body1 in numpy.ndenumerate(boundary1):
                    body2 = boundary2[z,y,x]
                    if body2 == 0 or body1 == 0:
                        continue
                    if body1 not in body2body[body2]:
                        body2body[body2][body1] = 0
                    body2body[body2][body1] += 1


            # create merge list 
            merge_list = []
            mutual_list = {}
            retired_list = set()

            small_overlap_prune = 0
            conservative_prune = 0
            aggressive_add = 0
            not_mutual = 0

            for body2, bodydict in body2body.items():
                if body2 in eligible_bodies:
                    bodysave = -1
                    max_val = hard_lb
                    total_val = 0
                    for body1, val in bodydict.items():
                        total_val += val
                        if val > max_val:
                            bodysave = body1
                            max_val = val
                    if bodysave == -1:
                        small_overlap_prune += 1
                    elif (stitch_mode == 1) and (max_val / float(total_val) < conservative_overlap):
                        conservative_prune += 1
                    elif (stitch_mode == 3) and (max_val / float(total_val) > conservative_overlap) and (max_val > liberal_lb):
                        merge_list.append([int(bodysave), int(body2)])
                        # do not add
                        retired_list.add((int(bodysave), int(body2))) 
                        aggressive_add += 1
                    else:
                        if int(bodysave) not in mutual_list:
                            mutual_list[int(bodysave)] = {}
                        mutual_list[int(bodysave)][int(body2)] = max_val
                       

            eligible_bodies = set(numpy.unique(boundary1[z1:z2, y1:y2, x1:x2]))
            body2body = {}
            
            if stitch_mode > 0:
                label1_bodies = numpy.unique(boundary1)
                for body in label1_bodies:
                    if body == 0:
                        continue
                    body2body[body] = {}

                # traverse volume to find maximum overlap
                for (z,y,x), body1 in numpy.ndenumerate(boundary1):
                    body2 = boundary2[z,y,x]
                    if body2 == 0 or body1 == 0:
                        continue
                    if body2 not in body2body[body1]:
                        body2body[body1][body2] = 0
                    body2body[body1][body2] += 1
            
            # add to merge list 
            for body1, bodydict in body2body.items():
                if body1 in eligible_bodies:
                    bodysave = -1
                    max_val = hard_lb
                    total_val = 0
                    for body2, val in bodydict.items():
                        total_val += val
                        if val > max_val:
                            bodysave = body2
                            max_val = val

                    if (int(body1), int(bodysave)) in retired_list:
                        # already in list
                        pass
                    elif bodysave == -1:
                        small_overlap_prune += 1
                    elif (stitch_mode == 1) and (max_val / float(total_val) < conservative_overlap):
                        conservative_prune += 1
                    elif (stitch_mode == 3) and (max_val / float(total_val) > conservative_overlap) and (max_val > liberal_lb):
                        merge_list.append([int(body1), int(bodysave)])
                        aggressive_add += 1
                    elif int(body1) in mutual_list:
                        partners = mutual_list[int(body1)]
                        if int(bodysave) in partners:
                            merge_list.append([int(body1), int(bodysave)])
                        else:
                            not_mutual += 1
                    else:
                        not_mutual += 1
            
            # handle offsets in mergelist
            offset1 = subvolume_offsets.value[subvolume1.roi_id] 
            offset2 = subvolume_offsets.value[subvolume2.roi_id] 
            for merger in merge_list:
                merger[0] = merger[0]+offset1
                merger[1] = merger[1]+offset2

            # return id and mappings, only relevant for stack one
            return (subvolume1.roi_id, merge_list)

        # key, mapping1; key mapping2 => key, mapping1+mapping2
        def reduce_mappings(b1, b2):
            b1.extend(b2)
            return b1

        # map from grouped boundary to substack id, mappings
        subvolume_mappings = grouped_boundaries.map(stitcher).reduceByKey(reduce_mappings)

        # reconcile all the mappings by sending them to the driver
        # (not a lot of data and compression will help but not sure if there is a better way)
        merge_list = []
        all_mappings = subvolume_mappings.collect()
        for (substack_id, mapping) in all_mappings:
            merge_list.extend(mapping)

        # make a body2body map
        body1body2 = {}
        body2body1 = {}
        for merger in merge_list:
            # body1 -> body2
            body1 = merger[0]
            if merger[0] in body1body2:
                body1 = body1body2[merger[0]]
            body2 = merger[1]
            if merger[1] in body1body2:
                body2 = body1body2[merger[1]]

            if body2 not in body2body1:
                body2body1[body2] = set()
            
            # add body1 to body2 map
            body2body1[body2].add(body1)
            # add body1 -> body2 mapping
            body1body2[body1] = body2

            if body1 in body2body1:
                for tbody in body2body1[body1]:
                    body2body1[body2].add(tbody)
                    body1body2[tbody] = body2

        body2body = zip(body1body2.keys(), body1body2.values())
       
        # potentially costly broadcast
        # (possible to split into substack to make more efficient but compression should help)
        master_merge_list = self.context.sc.broadcast(body2body)

        # use offset and mappings to relabel volume
        def relabel(key_label_mapping):
            import numpy

            (subvolume, label_chunk_c) = key_label_mapping
            labels = label_chunk_c.deserialize()

            # grab broadcast offset
            offset = subvolume_offsets.value[subvolume.roi_id]

            labels = labels + offset 
            # make sure 0 is 0
            labels[labels == offset] = 0

            # create default map 
            mapping_col = numpy.unique(labels)
            label_mappings = dict(zip(mapping_col, mapping_col))
           
            # create maps from merge list
            for mapping in master_merge_list.value:
                if mapping[0] in label_mappings:
                    label_mappings[mapping[0]] = mapping[1]

            # apply maps
            vectorized_relabel = numpy.frompyfunc(label_mappings.__getitem__, 1, 1)
            labels = vectorized_relabel(labels).astype(numpy.uint64)
       
            return (subvolume, CompressedNumpyArray(labels))

        # just map values with broadcast map
        # Potential TODO: consider fast join with partitioned map (not broadcast)
        return label_chunks.mapValues(relabel)


