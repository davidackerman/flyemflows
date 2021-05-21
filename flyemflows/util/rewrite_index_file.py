from struct import *
import numpy as np
from functools import cmp_to_key
import struct
import os

def _cmp_zorder(lhs, rhs) -> bool:
    def less_msb(x: int, y: int) -> bool:
        return x < y and x < (x ^ y)

    # Assume lhs and rhs array-like objects of indices.
    assert len(lhs) == len(rhs)
    # Will contain the most significant dimension.
    msd = 2
    # Loop over the other dimensions.
    for dim in [1, 0]:
        # Check if the current dimension is more significant
        # by comparing the most significant bits.
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] - rhs[msd]


def rewrite_index_with_empty_fragments(path):
    def get_fragment_positions_in_current_lod(fragment_positions_in_higher_lod):
            #get 8 subfragments
            fragment_positions_in_current_lod = []
            for fragment_position_in_higher_lod in fragment_positions_in_higher_lod:
                for x in range(2):
                    for y in range(2):
                        for z in range(2):
                            adjust = [x,y,z]
                            fragment_positions_in_current_lod.append(tuple([fragment_position_in_higher_lod[i]*2+adjust[i] for i in range(3)]))
            return set(fragment_positions_in_current_lod)
    
    def are_all_lower_lod_fragments_empty(lod,missing_fragment_position,fragments_to_delete):
        if lod==-1:
            return True
        else:
            current_lod_fragments = get_fragment_positions_in_current_lod(missing_fragment_position)
            if current_lod_fragments.issubset(all_missing_fragment_positions[lod]):
                fragments_to_delete[lod].update(current_lod_fragments)
                return are_all_lower_lod_fragments_empty(lod-1,current_lod_fragments,fragments_to_delete)
            else:
                fragments_to_delete = None
                return False

    def unpack_and_remove(datatype,num_elements,file_content):
        datatype = datatype*num_elements
        output = struct.unpack(datatype, file_content[0:4*num_elements])
        file_content = file_content[4*num_elements:] 
        return np.array(output),file_content
            

    with open(f"{path}.index", mode='rb') as file:
        file_content = file.read()
    
    chunk_shape,file_content = unpack_and_remove("f",3,file_content)
    grid_origin,file_content = unpack_and_remove("f",3,file_content)
    num_lods,file_content = unpack_and_remove("I",1,file_content)
    num_lods = num_lods[0]
    lod_scales,file_content = unpack_and_remove("f",num_lods,file_content)
    vertex_offsets,file_content = unpack_and_remove("I",num_lods*3,file_content)
    num_fragments_per_lod,file_content = unpack_and_remove("I",num_lods,file_content)
    original_fragment_positions = []
    all_current_fragment_positions = []
    all_current_fragment_offsets = []

    minimum_chunk_origin = np.array([1E9,1E9,1E9])
    maximum_chunk_origin = np.array([0,0,0])
    for lod in range(num_lods):
        fragment_positions,file_content = unpack_and_remove("I",num_fragments_per_lod[lod]*3,file_content)
        fragment_positions = fragment_positions.reshape((3,-1)).T
        minimum_chunk_origin = np.min( np.append(fragment_positions*chunk_shape*2**lod, [minimum_chunk_origin],axis=0), axis=0 )
        maximum_chunk_origin = np.max( np.append(fragment_positions*chunk_shape*2**lod, [maximum_chunk_origin], axis=0), axis=0 )
        fragment_offsets,file_content = unpack_and_remove("I",num_fragments_per_lod[lod],file_content)
        original_fragment_positions.append(list(map(tuple,fragment_positions)))
        all_current_fragment_positions.append(list(map(tuple,fragment_positions)))
        all_current_fragment_offsets.append(fragment_offsets.tolist())

    max_lod_chunk_shape = (chunk_shape*2**num_lods)
    chunking_start =  (max_lod_chunk_shape*minimum_chunk_origin//max_lod_chunk_shape).astype(int)
    chunking_end =  ((maximum_chunk_origin//max_lod_chunk_shape)*max_lod_chunk_shape+max_lod_chunk_shape).astype(int)
    all_missing_fragment_positions = []
    for lod in range(num_lods):
        current_lod_shape = (chunk_shape*2**lod).astype(int)
        x_range = range(chunking_start[0], chunking_end[0], current_lod_shape[0])
        y_range = range(chunking_start[1], chunking_end[1], current_lod_shape[1])
        z_range = range(chunking_start[2], chunking_end[2], current_lod_shape[2])
        all_required_fragment_positions = np.array([ [ [ [chunk_start_x/current_lod_shape[0], chunk_start_y/current_lod_shape[1], chunk_start_z//current_lod_shape[2]] for chunk_start_x in x_range] for chunk_start_y in y_range] for chunk_start_z in z_range]).reshape(-1,3).astype(int)
        all_required_fragment_positions = list(map(tuple,all_required_fragment_positions))
        current_missing_fragment_positions = set(all_required_fragment_positions).symmetric_difference(set(all_current_fragment_positions[lod]))
        all_missing_fragment_positions.append(current_missing_fragment_positions)

    #remove fragmenst that are zero across all lods
    fragments_to_delete = [set() for i in range(num_lods)]
    #is_fragment_empty_over_all_lods()
    for missing_fragment_positions in all_missing_fragment_positions[num_lods-1]: #start at largest lod
            missing_fragment_positions = set([missing_fragment_positions])
            current_fragments_to_delete = [set() for i in range(num_lods)]
            current_fragments_to_delete[num_lods-1].update(missing_fragment_positions)
            if are_all_lower_lod_fragments_empty(num_lods-2,missing_fragment_positions,current_fragments_to_delete):
                for lod in range(num_lods):
                    fragments_to_delete[lod].update(current_fragments_to_delete[lod])
                
    for lod in range(num_lods):
        all_missing_fragment_positions[lod]-=fragments_to_delete[lod]

    num_fragments_per_lod = []
    all_fragment_positions = []
    all_fragment_offsets = []
    for lod in range(num_lods):
        if len(all_missing_fragment_positions[lod])>0 :
            lod_fragment_positions = list(all_missing_fragment_positions[lod]) + all_current_fragment_positions[lod]
            lod_fragment_offsets = list( np.zeros(len(all_missing_fragment_positions[lod])) ) + all_current_fragment_offsets[lod]
        else:
            lod_fragment_positions = all_current_fragment_positions[lod]
            lod_fragment_offsets = all_current_fragment_offsets[lod]

        lod_fragment_offsets, lod_fragment_positions = zip(*sorted(zip(lod_fragment_offsets, lod_fragment_positions), key=cmp_to_key(lambda x, y: _cmp_zorder(x[1], y[1]))))
        all_fragment_positions.append(lod_fragment_positions)
        all_fragment_offsets.append(lod_fragment_offsets)
        num_fragments_per_lod.append(len(all_fragment_offsets[lod]))

    num_fragments_per_lod = np.array(num_fragments_per_lod)
    with open(f"{path}.index_with_empty_fragments", 'ab') as f:
        f.write(chunk_shape.astype('<f').tobytes())
        f.write(grid_origin.astype('<f').tobytes())

        f.write(struct.pack('<I', num_lods))
        f.write(lod_scales.astype('<f').tobytes())
        f.write(vertex_offsets.astype('<f').tobytes(order='C'))

        f.write(num_fragments_per_lod.astype('<I').tobytes())

        for lod in range(num_lods):
            fragment_positions = np.array(all_fragment_positions[lod]).reshape(-1,3)
            fragment_offsets = np.array(all_fragment_offsets[lod]).reshape(-1)

            f.write(fragment_positions.T.astype('<I').tobytes(order='C'))
            f.write(fragment_offsets.astype('<I').tobytes(order='C'))
        
        os.system(f"mv {path}.index_with_empty_fragments {path}.index")
