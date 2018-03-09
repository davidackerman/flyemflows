import sys
import argparse
import logging

import requests
from tqdm import tqdm # progress bar

import numpy as np
import pandas as pd

from dvidutils import LabelMapper

from DVIDSparkServices.util import Timer
from DVIDSparkServices.io_util.labelmap_utils import load_edge_csv

from DVIDSparkServices.dvid.metadata import DataInstance

# The labelindex_pb2 file was generated with the following command: 
# $ protoc --python_out=. labelindex.proto
from .labelindex_pb2 import LabelSetIndex

logger = logging.getLogger(__name__)

SUPERVOXEL_STATS_COLUMNS = ['segment_id', 'z', 'y', 'x', 'count']
AGGLO_MAP_COLUMNS = ['segment_id', 'body_id']

def gen_labelset_indexes(block_sv_stats_df, segment_to_body_df=None):
    assert list(block_sv_stats_df.columns) == SUPERVOXEL_STATS_COLUMNS

    if segment_to_body_df is None:
        # No agglomeration
        block_sv_stats_df['body_id'] = block_sv_stats_df['segment_id']
    else:
        assert list(segment_to_body_df.columns) == AGGLO_MAP_COLUMNS
        
        # This could be done via pandas merge(), followed by fillna(), etc.,
        # but I suspect LabelMapper is faster and more frugal with RAM.
        mapper = LabelMapper(segment_to_body_df['segment_id'].values, segment_to_body_df['body_id'].values)
        del segment_to_body_df
    
        block_sv_stats_df['body_id'] = 0
    
        chunk_size = 1_000_000
        for chunk_start in range(0, len(block_sv_stats_df), chunk_size):
            chunk_stop = min(chunk_start+chunk_size, len(block_sv_stats_df))
            chunk_segments = block_sv_stats_df[chunk_start:chunk_stop, 'segment_id'].values
            block_sv_stats_df[chunk_start:chunk_stop, 'body_id'] = mapper.apply(chunk_segments, allow_unmapped=True)

    for body_group_df in block_sv_stats_df.groupby('body_id'):
        body_id = body_group_df.iloc[0]['body_id']
        
        group_entries_total = 0
        label_set_index = LabelSetIndex()
        for block_df in body_group_df.groupby(['z', 'y', 'x']):
            coord_zyx = block_df.iloc[0][['z', 'y', 'x']].values.astype(np.int32)
            for sv, count in zip(block_df['segment_id'], block_df['count']):
                label_set_index.blocks[coord_zyx].counts[sv] = count

            group_entries_total += len(block_df)
        yield (body_id, label_set_index, group_entries_total)

def ingest_label_indexes(server, uuid, instance_name, block_sv_stats_df, segment_to_body_df=None, show_progress_bar=True):
    instance_info = DataInstance(server, uuid, instance_name)
    if instance_info.datatype != 'labelmap':
        raise RuntimeError(f"DVID instance is not a labelmap: {instance_name}")
    
    session = requests.Session()
    
    if not server.startswith('http://'):
        server = 'http://' + server

    with tqdm(len(block_sv_stats_df), disable=not show_progress_bar) as progress_bar:
        for body_id, label_set_index, group_entries_total in gen_labelset_indexes(block_sv_stats_df, segment_to_body_df):
            payload = label_set_index.SerializeToString()
            r = session.post(f'{server}/api/node/{uuid}/{instance_name}/index/{body_id}', data=payload)
            r.raise_for_status()
            progress_bar.update(group_entries_total)

def main():
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--agglomeration-mapping', '-m', required=False,
                        help='A CSV file with two columns, mapping supervoxels to agglomerated bodies. Any missing entries implicitly identity-mapped.')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('supervoxel_block_stats_csv',
                        help=f'A CSV file with columns: {SUPERVOXEL_STATS_COLUMNS}')

    args = parser.parse_args()

    segment_to_body_df = None
    if args.agglomeration_mapping:
        with Timer("Loading agglomeration mapping", logger):
            mapping_pairs = load_edge_csv(args.agglomeration_mapping)
            segment_to_body_df = pd.DataFrame(mapping_pairs, columns=AGGLO_MAP_COLUMNS)

    with Timer("Loading supervoxel block statistics file", logger):
        block_sv_stats_df = pd.read_csv(args.supervoxel_block_stats_csv, engine='c')

    with Timer(f"Grouping {len(block_sv_stats_df)} blockwise supervoxel counts and loading LabelSetIndexes", logger):
        ingest_label_indexes( args.server,
                              args.uuid,
                              args.labelmap_instance,
                              block_sv_stats_df,
                              segment_to_body_df,
                              True )

    logger.info(f"DONE.")

if __name__ == "__main__":
    main()
