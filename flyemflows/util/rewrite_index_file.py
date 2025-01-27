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


def rewrite_index_with_empty_fragments(path, mesh_boxes, current_lod_fragment_positions, current_lod_fragment_offsets):

    def unpack_and_remove(datatype,num_elements,file_content):
        datatype = datatype*num_elements
        output = struct.unpack(datatype, file_content[0:4*num_elements])
        file_content = file_content[4*num_elements:] 
        return np.array(output),file_content
            
    #index file contains info from all previous lods
    with open(f"{path}.index", mode='rb') as file:
        file_content = file.read()
    
    chunk_shape,file_content = unpack_and_remove("f",3,file_content)
    grid_origin,file_content = unpack_and_remove("f",3,file_content)
    num_lods,file_content = unpack_and_remove("I",1,file_content)
    num_lods = num_lods[0]
    lod_scales,file_content = unpack_and_remove("f",num_lods,file_content)
    vertex_offsets,file_content = unpack_and_remove("I",num_lods*3,file_content)
    num_fragments_per_lod,file_content = unpack_and_remove("I",num_lods,file_content)
    all_current_fragment_positions = []
    all_current_fragment_offsets = []

    for lod in range(num_lods):
        fragment_positions,file_content = unpack_and_remove("I",num_fragments_per_lod[lod]*3,file_content)
        fragment_positions = fragment_positions.reshape((3,-1)).T
        fragment_offsets,file_content = unpack_and_remove("I",num_fragments_per_lod[lod],file_content)
        all_current_fragment_positions.append(fragment_positions.astype(int))
        all_current_fragment_offsets.append(fragment_offsets.tolist())

    #now we are going to add the new lod info and update lower lods
    num_lods+=1
    all_current_fragment_positions.append(current_lod_fragment_positions.astype(int))
    all_current_fragment_offsets.append(current_lod_fragment_offsets.tolist())

    #first process based on newly added fragments
    all_missing_fragment_positions = []
    for lod in range(num_lods):
        all_required_fragment_positions = set()
        current_chunk_shape = (chunk_shape*2**lod).astype(int)

        if lod==num_lods-1: #then we are processing newest lod
            #add those that are required based on lower lods
            for lower_lod in range(lod):
                all_required_fragment_positions_np = np.unique(all_current_fragment_positions[lower_lod]//2**(lod-lower_lod), axis=0).astype(int) 
                all_required_fragment_positions.update( set(map(tuple,all_required_fragment_positions_np)) )
        else:
            # update lower lods based on current lod
            for mesh_box in mesh_boxes:
                #normally we would just do the following with -0 and +1, but because of quantization that occurs(?), this makes things extra conservative so we don't miss things
                chunking_start =  np.maximum((mesh_box[0]*1.0)//current_chunk_shape-1, [0,0,0]).astype(np.uint32) #ensures that it is positive, otherwise wound up with -1 to uint, causing errors
                chunking_end =  (mesh_box[1]//current_chunk_shape)+2
                x_range = range(chunking_start[0], chunking_end[0])
                y_range = range(chunking_start[1], chunking_end[1])
                z_range = range(chunking_start[2], chunking_end[2])
                all_required_fragment_positions_np = np.array([ [ [ [chunk_start_x, chunk_start_y, chunk_start_z] for chunk_start_x in x_range] for chunk_start_y in y_range] for chunk_start_z in z_range]).reshape(-1,3).astype(int)
                all_required_fragment_positions.update( set(map(tuple,all_required_fragment_positions_np)) )
        current_missing_fragment_positions = all_required_fragment_positions - set(map(tuple,all_current_fragment_positions[lod])) 
        all_missing_fragment_positions.append(current_missing_fragment_positions)

    num_fragments_per_lod = []
    all_fragment_positions = []
    all_fragment_offsets = []
    for lod in range(num_lods):
        if len(all_missing_fragment_positions[lod])>0 :
            lod_fragment_positions = list(all_missing_fragment_positions[lod]) + list(all_current_fragment_positions[lod])
            lod_fragment_offsets = list( np.zeros(len(all_missing_fragment_positions[lod])) ) + all_current_fragment_offsets[lod]
        else:
            lod_fragment_positions = all_current_fragment_positions[lod]
            lod_fragment_offsets = all_current_fragment_offsets[lod]

        lod_fragment_offsets, lod_fragment_positions = zip(*sorted(zip(lod_fragment_offsets, lod_fragment_positions), key=cmp_to_key(lambda x, y: _cmp_zorder(x[1], y[1]))))
        all_fragment_positions.append(lod_fragment_positions)
        all_fragment_offsets.append(lod_fragment_offsets)
        num_fragments_per_lod.append(len(all_fragment_offsets[lod]))

    num_fragments_per_lod = np.array(num_fragments_per_lod)
    lod_scales = np.array([2**i for i in range(num_lods)])
    vertex_offsets = np.array([[0.,0.,0.] for _ in range(num_lods)])
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