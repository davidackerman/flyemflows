import os
import copy
import logging
from itertools import chain

import numpy as np
import pandas as pd

import dask.bag as db
import dask.dataframe as ddf

from neuclease.util import Timer, SparseBlockMask, box_intersection, extract_subvol
from neuclease.dvid import (fetch_mappings, fetch_repo_instances, create_tarsupervoxel_instance,
                            create_instance, is_locked, post_load, post_keyvalues, fetch_exists, fetch_keys,
                            fetch_supervoxels, fetch_server_info, fetch_mapping)
from dvid_resource_manager.client import ResourceManagerClient
from dvidutils import LabelMapper

from vol2mesh import Mesh

from ..util.dask_util import drop_empty_partitions
from .util.config_helpers import BodyListSchema, load_body_list
from ..volumes import VolumeService, SegmentationVolumeSchema, DvidVolumeService
from ..brick import BrickWall
from . import Workflow

logger = logging.getLogger(__name__)

class CreateMeshes(Workflow):
    """
    Generate meshes for many (or all) segments in a volume.
    """
    GenericDvidInstanceSchema = \
    {
        "description": "Parameters to specify a generic dvid instance (server/uuid/instance).\n"
                       "Omitted values will be copied from the input, or given default values.",
        "type": "object",
        "required": ["server", "uuid"],
    
        #"default": {}, # Must not have default
        "additionalProperties": False,
        "properties": {
            "server": {
                "description": "location of DVID server to READ.",
                "type": "string",
                "default": ""
            },
            "uuid": {
                "description": "version node from dvid",
                "type": "string",
                "default": ""
            },
            "instance": {
                "description": "Name of the instance to create",
                "type": "string",
                "default": ""
            },
            "sync-to": {
                "description": "When creating a tarsupervoxels instance, it should be sync'd to a labelmap instance.\n"
                               "Give the instance name here.",
                "type": "string",
                "default": ""
            },
            "create-if-necessary": {
                "description": "Whether or not to create the instance if it doesn't already exist.\n"
                               "If you expect the instance to exist on the server already, leave this\n"
                               "set to False to avoid confusion in the case of typos, UUID mismatches, etc.\n",
                "type": "boolean",
                "default": False
            },
        }
    }

    TarsupervoxelsOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "tarsupervoxels": GenericDvidInstanceSchema
        }
    }

    KeyvalueOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "keyvalue": GenericDvidInstanceSchema
        }
    }

    DirectoryOutputSchema = \
    {
        "additionalProperties": False,
        "properties": {
            "directory": {
                "description": "Directory to write supervoxel meshes into.",
                "type": "string",
                "default": "meshes"
            }
        }
    }

    MeshParametersSchema = \
    {
        # TODO: skip-decimation-body-size
        # TODO: downsample-before-marching-cubes?
        "properties": {
            "smoothing": {
                "description": "How many iterations of smoothing to apply to each mesh before decimation.",
                "type": "integer",
                "default": 0
            },
            "decimation": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "To disable decimation, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0, # 1.0 == disable
                "default": 1.0
            },
            "compute-normals": {
                "description": "Compute vertex normals and include them in the uploaded results.",
                "type": "boolean",
                "default": False
            }
        }
    }

    SizeFiltersSchema = \
    {
        "properties": {
            "minimum-supervoxel-size": {
                "type": "number",
                "default": 0
            },
            "maximum-supervoxel-size": {
                "type": "number",
                "default": 1e12
            },
            "minimum-body-size": {
                "type": "number",
                "default": 0
            },
            "maximum-body-size": {
                "type": "number",
                "default": 1e12
            }
        }
    }

    SupervoxelListSchema = copy.copy(BodyListSchema)
    SupervoxelListSchema["description"] = \
        ("List of supervoxel IDs to process, or a path to a CSV file with the list.\n"
         "NOTE: If you're using a non-labelmap source (e.g. HDF5, etc.),\n"
         "      it is considered supervoxel data.\n")

    BodyListSchema = copy.copy(BodyListSchema)
    BodyListSchema["description"] = \
        ("List of body IDs to process, or a path to a CSV file with the list.\n"
         "NOTE: If you're using a non-labelmap source (e.g. HDF5, etc.), \n"
         "      it is considered supervoxel data.\n"
         "      This config setting can only be used when using a labelmap source.\n")

    CreateMeshesOptionsSchema = \
    {
        "type": "object",
        "description": "Settings specific to the CreateMeshes workflow",
        "default": {},
        "additionalProperties": False,
        "properties": {
            "subset-supervoxels": SupervoxelListSchema,
            "subset-bodies": BodyListSchema,

            "halo": {
                "description": "How much overlapping context between bricks in the grid (in voxels)\n",
                "type": "integer",
                "minValue": 1,
                "default": 0
            },

            "pre-stitch-parameters": MeshParametersSchema,
            "post-stitch-parameters": MeshParametersSchema,

            "stitch-method": {
                "description": "How to combine each segment's blockwise meshes into a single file.\n"
                               "Choices are 'simple-concatenate' and 'stitch'.\n"
                               "The 'stitch' method should only be used when halo >= 1 and there is no pre-stitch smoothing or decimation.\n",
                "type": "string",
                "enum": ["simple-concatenate", # Just dump the vertices and faces into the same file
                                               # (renumber the faces to match the vertices, but don't unify identical vertices.)
                                               # If using this setting it is important to use a task-block-halo of > 2 to hide
                                               # the seams, even if smoothing is used.

                         "stitch",             # Search for duplicate vertices and remap the corresponding face corners,
                                               # so that the duplicate entries are not used. Topologically stitches adjacent faces.
                                               # Will be ineffective unless you used a task-block-halo of at least 1, and no
                                               # pre-stitch smoothing or decimation.
                        ],
                
                "default": "simple-concatenate",
            },
            "size-filters": SizeFiltersSchema,
            
            "max-body-vertices": {
                "description": "If necessary, dynamically increase decimation on a per-body, per-brick basis so that\n"
                               "the total vertex count for each mesh (across all bricks) final mesh will not exceed\n"
                               "this total vertex count.\n"
                               "If omitted, no maximum is used.\n",
                "oneOf": [{"type": "number"}, {"type": "null"}],
                "default": None
            },

            "rescale-before-write": {
                "description": "How much to rescale the meshes before writing to DVID.\n"
                               "Specified as a multiplier, not power-of-2 'scale'.\n",
                "type": "number",
                "default": 1.0
            },
            "format": {
                "description": "Format to save the meshes in. ",
                "type": "string",
                "enum": ["obj",     # Wavefront OBJ (.obj)
                         "drc",     # Draco (compressed) (.drc)
                         "ngmesh"], # "neuroglancer mesh" format -- a custom binary format.  Note: Data is presumed to be 8nm resolution
                "default": "obj"
            },
            "include-empty": {
                "description": "Objects too small to generate proper meshes for may be 'serialized' as an empty buffer (0 bytes long).\n"
                               "This setting specifies whether 0-byte files are uploaded to the destination server in such cases,\n"
                               "or if they are omitted entirely.\n",
                "type": "boolean",
                "default": False
            },
            "skip-existing": {
                "description": "Do not generate meshes for meshes that already exist in the output location.\n",
                "type": "boolean",
                "default": False
            },
            
        }
    }

    Schema = copy.deepcopy(Workflow.schema())
    Schema["properties"].update({
        "input": SegmentationVolumeSchema,

        "output": {
            "oneOf": [
                DirectoryOutputSchema,
                TarsupervoxelsOutputSchema,
                KeyvalueOutputSchema,
            ],
            "default": {"directory": "meshes"}
        },

        "createmeshes": CreateMeshesOptionsSchema,
        
    })

    @classmethod
    def schema(cls):
        return CreateMeshes.Schema

    def _sanitize_config(self):
        options = self.config["createmeshes"]
        if options['stitch-method'] == 'stitch' and options['halo'] != 1:
            logger.warn("Your config uses 'stitch' aggregation, but your halo != 1.\n"
                        "This will waste CPU and/or lead to unintuitive results.")

    def _init_input(self): 
        input_config = self.config["input"]
        resource_config = self.config["resource-manager"]
        self.resource_mgr_client = ResourceManagerClient(resource_config["server"], resource_config["port"])
        input_service = VolumeService.create_from_config(input_config, self.resource_mgr_client)
        self.input_service = input_service


    def _prepare_output(self):
        output_cfg = self.config["output"]
        output_fmt = self.config["createmeshes"]["format"]

        ## directory output
        if 'directory' in output_cfg:
            os.makedirs(output_cfg['directory'])
            return

        ##
        ## DVID output (either keyvalue or tarsupervoxels)
        ##
        (instance_type,) = output_cfg.keys()

        server = output_cfg[instance_type]['server']
        uuid = output_cfg[instance_type]['uuid']
        instance = output_cfg[instance_type]['uuid']

        if is_locked(server, uuid):
            info = fetch_server_info(server)
            if "Mode" in info and info["Mode"] == "allow writes on committed nodes":
                logger.warn(f"Output is a locked node ({uuid}), but server is in full-write mode.")
            else:
                raise RuntimeError(f"Can't write to node {uuid} because it is locked.")

        if instance_type == 'tarsupervoxels' and not self.input_is_labelmap_supervoxels():
            msg = ("You shouldn't write to a tarsupervoxels instance unless "
                   "you're reading supervoxels from a labelmap input.\n"
                   "Use a labelmap input source, and set supervoxels: true")
            raise RuntimeError(msg)

        existing_instances = fetch_repo_instances(self.server, self.uuid)
        if instance in existing_instances:
            # Instance exists -- nothing to do.
            return

        if not output_cfg[instance_type]['create-if-necessary']:
            msg = (f"Output instance '{instance}' does not exist, "
                   "and your config did not specify create-if-necessary")
            raise RuntimeError(msg)

        assert instance_type in ('tarsupervoxels', 'keyvalue')
        
        ## keyvalue output
        if instance_type == "keyvalue":
            create_instance(server, uuid, instance, "keyvalue", tags=["type=meshes"])
            return

        ## tarsupervoxels output
        sync_instance = output_cfg["tarsupervoxels"]["sync-to"]
        if not sync_instance:
            msg = ("Can't create a tarsupervoxels instance unless "
                   "you specify a 'sync-to' labelmap instance name.")
            raise RuntimeError(msg)

        if sync_instance not in existing_instances:
            msg = ("Can't sync to labelmap instance '{sync_instance}': "
                   "it doesn't exist on the output server.")
            raise RuntimeError(msg)

        create_tarsupervoxel_instance(server, uuid, instance, sync_instance, output_fmt, tags=["type=meshes"])



    def input_is_labelmap_supervoxels(self):
        if isinstance(self.input_service.base_service, DvidVolumeService):
            return self.input_service.base_service.supervoxels
        return False


    def input_is_labelmap_bodies(self):
        if isinstance(self.input_service.base_service, DvidVolumeService):
            return not self.input_service.base_service.supervoxels
        return False

    def execute(self):
        """
        This workflow is designed for generating supervoxel meshes from a labelmap input (with supervoxels: true).
        But other sources. In those cases, each object is treated as a 'supervoxel', and each 'body' has only one supervoxel.
        NOTE:
            If your input is a labelmap, but you have configured it to read body labels (i.e. supervoxels: false),
            then it is treated like any other non-supervoxel-aware datasource, such as HDF5.
            That is, in the code below, 'supervoxels' and 'bodies' refer to the same thing.
            The real underlying supervoxel IDs in the labelmap instance are not used.
        """
        self._sanitize_config()
        options = self.config["createmeshes"]

        self._init_input()
        self._prepare_output()        

        subset_supervoxels = load_body_list(options["subset-supervoxels"], self.input_is_labelmap_supervoxels())
        subset_bodies = load_body_list(options["subset-bodies"], self.input_is_labelmap_supervoxels())

        if len(subset_supervoxels) and len(subset_bodies):
            raise RuntimeError("Can't use both subset-supervoxels and subset-bodies.  Choose one.")

        if self.input_is_labelmap_supervoxels():
            if subset_bodies:
                with Timer("Fetching supervoxel set for labelmap bodies", logger):
                    def fetch_svs(body):
                        return fetch_supervoxels(*self.input_service.instance_triple, body)
                    svs = db.from_sequence(subset_bodies, npartitions=512).map(fetch_svs).compute()
                    subset_supervoxels = chain(*svs)

        brickwall = self.init_brickwall(self.input_service, subset_supervoxels)

        if self.input_is_labelmap_bodies():
            assert len(subset_supervoxels) == 0, \
                "Can't use subset-supervoxels when reading from a labelmap in body mode.  Please use subset-bodies."

            # In the code below, voxels read from the input source are referred to as 'supervoxels'.
            # Since the user is reading pre-mapped bodies from a labelmap,
            # We won't be computing group-stats like we would with supervoxel meshes.
            # We now refer to the body IDs as if they were supervoxel IDs.
            #
            # FIXME: I need to just rename 'supervoxel' to 'label' and make this less confusing.
            subset_supervoxels = subset_bodies
            
        bricks_ddf = BrickWall.bricks_as_ddf(brickwall.bricks, logical=True)
        bricks_ddf = bricks_ddf[['lz0', 'ly0', 'lx0', 'brick']]
        
        def compute_brick_labelcounts(brick_df):
            brick_counts_dfs = []
            for row in brick_df.itertuples():
                brick = row.brick
                inner_box = box_intersection(brick.logical_box, brick.physical_box)
                inner_box -= brick.physical_box[0]
                inner_vol = extract_subvol(brick.volume, inner_box)
                label_counts = pd.Series(inner_vol.reshape(-1)).value_counts().sort_index()
                label_counts.index.name = 'label'
                label_counts.name = 'count'
                if label_counts.index[0] == 0:
                    label_counts = label_counts.iloc[1:]
                
                brick_counts_df = label_counts.reset_index()
                brick_counts_df['lz0'] = brick.logical_box[0,0]
                brick_counts_df['ly0'] = brick.logical_box[0,1]
                brick_counts_df['lx0'] = brick.logical_box[0,2]
                brick_counts_dfs.append(brick_counts_df)

            return pd.concat(brick_counts_dfs, ignore_index=True)

        dtypes = {'label': np.uint64, 'count': np.int64,
                  'lz0': np.int32, 'ly0': np.int32, 'lx0': np.int32}
        brick_counts_df = bricks_ddf.map_partitions(compute_brick_labelcounts, meta=dtypes).clear_divisions().compute()
        
        if self.input_is_labelmap_supervoxels():
            seg_instance = self.input_service.base_service.instance_triple

            brick_counts_df['sv'] = brick_counts_df['label'].values

            # Arbitrary heuristic for whether to do the body-lookups on DVID or on the client.
            if len(brick_counts_df['sv']) < 100_000:
                # If we're only dealing with a few supervoxels,
                # ask dvid to map them to bodies for us.
                brick_counts_df['body'] = fetch_mapping(*seg_instance, brick_counts_df['sv']).values
            else:
                # If we're dealing with a lot of supervoxels, ask for
                # the entire mapping, and look up the bodies ourselves.
                mapping = fetch_mappings(*seg_instance)
                mapper = LabelMapper(mapping.index.values, mapping.values)
                brick_counts_df['body'] = mapper.apply(brick_counts_df['sv'].values)
            
            total_sv_counts = brick_counts_df.groupby('sv')['count'].sum().rename('sv_size').reset_index()
            total_body_counts = brick_counts_df.groupby('body')['count'].sum().rename('body_size').reset_index()
        else:
            # Every label is treated as a supervoxel for our purposes.
            brick_counts_df['sv'] = brick_counts_df['label']
            brick_counts_df['body'] = brick_counts_df['label']
            total_sv_counts = brick_counts_df.groupby('sv')['count'].sum().rename('sv_size').reset_index()
            total_body_counts = total_sv_counts.rename(columns={'sv': 'body', 'sv_size': 'body_size'})

        brick_counts_df = brick_counts_df.merge(total_sv_counts, 'left', 'sv')
        brick_counts_df = brick_counts_df.merge(total_body_counts, 'left', 'body')

        logger.info("Saving brick-counts.npy")
        np.save('brick-counts.npy', brick_counts_df.to_records(index=False))

        # Filter for subset        
        if subset_supervoxels:
            sv_set = set(subset_supervoxels) #@UnusedVariable
            brick_counts_df = brick_counts_df.query('sv in @sv_set')

        # Filter for size
        size_filters = options["size-filters"]
        min_sv_size = size_filters['minimum-supervoxel-size'] #@UnusedVariable
        max_sv_size = size_filters['maximum-supervoxel-size'] #@UnusedVariable
        min_body_size = size_filters['minimum-body-size']     #@UnusedVariable
        max_body_size = size_filters['maximum-body-size']     #@UnusedVariable
        q = ('sv_size >= @min_sv_size and sv_size <= @max_sv_size and '
             'body_size >= @min_body_size and body_size <= @max_body_size')
        brick_counts_df = brick_counts_df.query(q)
        
        # Filter for already existing
        if options["skip-existing"]:
            with Timer("Determining which meshes are already stored (skip-existing)", logger):
                fmt = options["format"]
                destination = self.config["output"]
                (destination_type,) = destination.keys()
                assert destination_type in ('directory', 'keyvalue', 'tarsupervoxels')
    
                all_svs = pd.unique(brick_counts_df['sv'])
                if destination_type == 'directory':
                    d = self.config["output"]["directory"]
                    existing_svs = set()
                    for sv in all_svs:
                        if os.path.exists(f"{d}/{sv}.{fmt}"):
                            existing_svs.add(sv)
                elif destination_type == 'tarsupervoxels':
                    tsv_instance = [destination['tarsupervoxels'][k] for k in ('server', 'uuid', 'instance')]
                    exists = fetch_exists(*tsv_instance, all_svs)
                    existing_svs = set(exists[exists].index)
                elif destination_type == 'keyvalue':
                    logger.warning("Using skip-exists with a keyvalue output.  This might take a LONG time if there are many meshes already stored.")
                    kv_instance = [destination['keyvalue'][k] for k in ('server', 'uuid', 'instance')]
                    keys = fetch_keys(*kv_instance)
                    existing_svs = set(int(k[:-4]) for k in keys)
            
            brick_counts_df = brick_counts_df.query('sv not in @existing_svs')

        brick_counts_grouped_df = brick_counts_df.groupby(['lz0', 'ly0', 'lx0'])[['sv', 'sv_size', 'body', 'body_size']].agg(list).reset_index()

        with Timer("Distributing counts to bricks", logger):
            # Send count lists to their respective bricks
            # Use an inner merge to discard bricks that had no objects of interest.
            brick_counts_grouped_ddf = ddf.from_pandas(brick_counts_grouped_df, npartitions=1) # FIXME: What's good here?
            bricks_ddf = bricks_ddf.merge(brick_counts_grouped_ddf, 'inner', ['lz0', 'ly0', 'lx0'])
            bricks_ddf = drop_empty_partitions(bricks_ddf)
        
        def compute_meshes_for_bricks(bricks_partition_df):
            assert len(bricks_partition_df) > 0, "partition is empty" # drop_empty_partitions() should have eliminated these.
            result_dfs = []
            for row in bricks_partition_df.itertuples():
                stats_df = pd.DataFrame({'sv': row.sv, 'sv_size': row.sv_size,
                                         'body': row.body, 'body_size': row.body_size})

                brick_meshes_df = compute_meshes_for_brick(row.brick, stats_df, options)
                brick_meshes_df['lz0'] = row.lz0
                brick_meshes_df['ly0'] = row.ly0
                brick_meshes_df['lx0'] = row.lx0

                # Reorder columns
                cols = ['lz0', 'ly0', 'lx0', 'sv', 'body', 'mesh', 'vertex_count', 'compressed_size']
                brick_meshes_df = brick_meshes_df[cols]
                
                result_dfs.append(brick_meshes_df)

            return pd.concat(result_dfs, ignore_index=True)
                
        dtypes = {'lz0': np.int32, 'ly0': np.int32, 'lx0': np.int32,
                  'sv': np.int64, 'body': np.int64,
                  'mesh': object,
                  'vertex_count': int, 'compressed_size': int}

        brick_meshes_ddf = bricks_ddf.map_partitions(compute_meshes_for_bricks, meta=dtypes).clear_divisions()

        # Export brick mesh statistics
        os.makedirs('brick-mesh-stats')
        brick_stats_ddf = brick_meshes_ddf[['sv', 'body', 'vertex_count', 'compressed_size']]
        brick_stats_ddf.to_csv('brick-mesh-stats/partition-*.csv', index=False, header=True)
        del brick_stats_ddf

        final_smoothing = options["post-stitch-parameters"]["smoothing"]
        final_decimation = options["post-stitch-parameters"]["decimation"]
        compute_normals = options["post-stitch-parameters"]["compute-normals"]

        stitch_method = options["stitch-method"]
        def assemble_sv_meshes(sv_brick_meshes_df):
            sv = sv_brick_meshes_df['sv'].iloc[0]

            mesh = Mesh.concatenate_meshes(sv_brick_meshes_df['mesh'])
            if stitch_method == 'stitch':
                mesh.stitch_adjacent_faces(drop_unused_vertices=True, drop_duplicate_faces=True)
            
            if final_smoothing != 0:
                mesh.laplacian_smooth(final_smoothing)
            
            if final_decimation != 1.0:
                mesh.simplify(final_decimation, in_memory=True)
            
            if not compute_normals:
                mesh.drop_normals()
            elif len(mesh.normals_zyx) == 0:
                mesh.recompute_normals()

            vertex_count = len(mesh.vertices_zyx)
            compressed_size = mesh.compress()
            
            return pd.DataFrame({'sv': [sv],
                                 'mesh': [mesh],
                                 'vertex_count': [vertex_count],
                                 'compressed_size': compressed_size})

        sv_brick_meshes_ddf = brick_meshes_ddf.groupby('sv')
        
        dtypes = {'sv': np.uint64, 'mesh': object, 'vertex_count': np.int64, 'compressed_size': int}
        sv_meshes_ddf = sv_brick_meshes_ddf.apply(assemble_sv_meshes, meta=dtypes)

        # Export stitched mesh statistics        
        os.makedirs('stitched-mesh-stats')
        sv_meshes_ddf[['sv', 'vertex_count', 'compressed_size']].to_csv('stitched-mesh-stats/partition-*.csv', index=False, header=True)
        
        # TODO: Repartition?
        
        destination = self.config["output"]
        fmt = options["format"]
        include_empty = options["include-empty"]
        resource_mgr = self.resource_mgr_client
        def write_sv_meshes(sv_meshes_df):
            (destination_type,) = destination.keys()
            assert destination_type in ('directory', 'keyvalue', 'tarsupervoxels')

            names = [f"{sv}.{fmt}" for sv in sv_meshes_df['sv']]
            binary_meshes = [serialize_mesh(sv, mesh, None, fmt=fmt)
                             for (sv, mesh) in sv_meshes_df[['sv', 'mesh']].itertuples(index=False)]
            keyvalues = dict(zip(names, binary_meshes))
            filesizes = [len(mesh_bytes) for mesh_bytes in keyvalues.values()]
            
            if not include_empty:
                keyvalues = {k:v for (k,v) in keyvalues.items() if len(v) > 0}

            if destination_type == 'directory':
                for name, mesh_bytes in keyvalues.items():
                    path = destination['directory'] + "/" + name
                    with open(path, 'wb') as f:
                        f.write(mesh_bytes)
            else:
                instance = (destination[destination_type][k] for k in ('server', 'uuid', 'instance'))
                with resource_mgr.access_context(instance[0], False, 1, sum(filesizes)):
                    if destination_type == 'tarsupervoxels':
                        post_load(*instance, keyvalues)
                    elif 'keyvalue' in destination:
                        post_keyvalues(*instance, keyvalues)

            result_df = sv_meshes_df[['sv', 'vertex_count', 'compressed_size']]
            result_df['file_size'] = filesizes
            return result_df
        
        dtypes = {'sv': np.uint64, 'vertex_count': np.int64, 'compressed_size': int, 'file_size': int}
        final_stats_df = sv_meshes_ddf.map_partitions(write_sv_meshes, meta=dtypes).clear_divisions().compute()
        np.save('final-mesh-stats.npy', final_stats_df.to_records(index=False))
        

    def init_brickwall(self, volume_service, subset_labels):
        sbm = None
        if subset_labels:
            try:
                brick_coords_df = volume_service.sparse_block_mask_for_labels(subset_labels)
                np.save('brick-coords.npy', brick_coords_df.to_records(index=False))
    
                brick_shape = volume_service.preferred_message_shape
                brick_indexes = brick_coords_df[['z', 'y', 'x']].values // brick_shape
                sbm = SparseBlockMask.create_from_lowres_coords(brick_indexes, brick_shape)
            except NotImplementedError:
                logger.warning("The volume service does not support sparse fetching.  All bricks will be analyzed.")
                sbm = None
            
        with Timer("Initializing BrickWall", logger):
            # Aim for 2 GB RDD partitions when loading segmentation
            GB = 2**30
            target_partition_size_voxels = 2 * GB // np.uint64().nbytes
            
            # Apply halo WHILE downloading the data.
            # TODO: Allow the user to configure whether or not the halo should
            #       be fetched from the outset, or added after the blocks are loaded.
            halo = self.config["createmeshes"]["halo"]
            brickwall = BrickWall.from_volume_service(volume_service, 0, None, self.client, target_partition_size_voxels, halo, sbm, compression='lz4_2x')

        return brickwall


def serialize_mesh(sv, mesh, path=None, fmt=None):
    """
    Call mesh.serialize(), but if an error occurs,
    log it and save an .obj to 'bad-meshes'
    """
    try:
        return mesh.serialize(path, fmt)
    except:
        if not os.path.exists('bad-meshes'):
            os.makedirs('bad-meshes', exist_ok=True)
        output_path = f'bad-meshes/failed-serialization-{sv}.obj'
        mesh.serialize(output_path, 'obj')
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to serialize mesh.  Wrote to {output_path}")
        return b''


def compute_meshes_for_brick(brick, stats_df, options):
    smoothing = options["pre-stitch-parameters"]["smoothing"]
    decimation = options["pre-stitch-parameters"]["decimation"]
    rescale_factor = options["rescale-before-write"]

    # TODO: max-body-vertices

    cols = ['sv', 'body', 'mesh', 'vertex_count', 'compressed_size']
    if len(stats_df) == 0:
        empty64 = np.zeros((0,1), dtype=np.uint64)
        emptyObject = np.zeros((0,1), dtype=object)
        return pd.DataFrame([empty64, empty64, emptyObject, empty64, empty64])
    
    volume = brick.volume
    brick.compress()
    
    meshes = []
    for row in stats_df.itertuples():
        mesh, vertex_count, compressed_size = generate_mesh(volume, brick.physical_box, row.sv,
                                                            smoothing, decimation, rescale_factor)
        meshes.append( (row.sv, row.body, mesh, vertex_count, compressed_size) )
    
    return pd.DataFrame(meshes, columns=cols)


def generate_mesh(volume, box, label, smoothing, decimation, rescale_factor):
    mask = (volume == label)
    mesh = Mesh.from_binary_vol(mask, box)
    
    if smoothing != 0:
        mesh.laplacian_smooth(smoothing)
    
    # Don't bother decimating really tiny meshes -- something usually goes wrong anyway.
    if decimation != 1.0 and len(mesh.vertices_zyx) > 10:
        # TODO: Implement a timeout here for the in-memory case (use multiprocessing)?
        mesh.simplify(decimation, in_memory=True)
    
    if rescale_factor != 1.0:
        mesh.vertices_zyx[:] *= rescale_factor
    
    vertex_count = len(mesh.vertices_zyx)
    compressed_size = mesh.compress()

    return mesh, vertex_count, compressed_size

