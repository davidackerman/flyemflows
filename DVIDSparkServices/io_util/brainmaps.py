import os
import json
from itertools import chain

from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials

import numpy as np
import snappy

# DEBUG 'logging' (Doesn't actually use logging module)
#import httplib2
#httplib2.debuglevel = 1

BRAINMAPS_API_VERSION = 'v1'
BRAINMAPS_BASE_URL = f'https://brainmaps.googleapis.com/{BRAINMAPS_API_VERSION}'


class BrainMapsVolume:
    def __init__(self, project, dataset, volume_id, change_stack_id="", dtype=None, skip_checks=False):
        """
        Utility for accessing subvolumes of a BrainMaps volume.
        Instances of this class are pickleable, but they will have to re-authenticate after unpickling.
        
        For REST API details, see the BrainMaps API documentation:
        https://developers.google.com/brainmaps/v1/rest/
        
        (To access the docs, you need to email a BrainMaps developer at Google and
        ask them to add your email to brainmaps-tt@googlegroups.com.)
        
        Args:
            project, dataset, volume_id, and (optionally) change_stack_id can be extracted from a brainmaps volume url:
            
            >>> url = 'brainmaps://274750196357:janelia-flyem-cx-flattened-tabs:sec26_seg_v2a:ffn_agglo_pass1_seg5663627_medt160'
            >>> full_id = url.split('://')[1]
            >>> project, dataset, volume_id, change_stack_id = full_id.split(':')

            dtype: (Optional.) If not provided, a separate request to the 
                   BrainMaps API will be made to determine the volume voxel type.

            skip_checks: If True, verify that the volume_id and change_stack_id exist on the server.
                         Otherwise, skip those checks, to minimize overhead.
        """
        self.project = project
        self.dataset = dataset
        self.volume_id = volume_id
        self.change_stack_id = change_stack_id
        self.skip_checks = skip_checks
        self._dtype = None # Assigned *after* check below.

        # These members are lazily computed/memoized.
        self._http = None
        self._bounding_boxes = None
        self._geometries = None

        if not self.skip_checks:
            volume_list = fetch_json(self.http, f'{BRAINMAPS_BASE_URL}/volumes')
            if f'{project}:{dataset}:{volume_id}' not in volume_list['volumeId']:
                raise RuntimeError(f"BrainMaps volume does not exist on server: {project}:{dataset}:{volume_id}")
    
            if change_stack_id:
                if change_stack_id not in self.get_change_stacks():
                    raise RuntimeError(f"ChangeStackId doesn't exist on the server: '{change_stack_id}'")

            if dtype:
                assert self.dtype == dtype, \
                    f"Provided dtype {dtype} doesn't match volume metadata ({self.dtype})"

        self._dtype = dtype


    def get_subvolume(self, box, scale=0):
        """
        Fetch a subvolume from the remote BrainMaps volume.
        
        Args:
            box: (start, stop) tuple, in ZYX order.
            scale: Which scale to fetch the subvolume from.
        
        Returns:
            volume (ndarray), where volume.shape = (stop - start)
        """
        box = np.asarray(box)
        corner_zyx = box[0]
        shape_zyx = box[1] - box[0]
        
        corner_xyz = corner_zyx[::-1]
        shape_xyz = shape_zyx[::-1]
        
        snappy_data = fetch_subvol_data( self.http,
                                         self.project,
                                         self.dataset,
                                         self.volume_id,
                                         corner_xyz,
                                         shape_xyz,
                                         scale,
                                         self.change_stack_id,
                                         subvol_format='RAW_SNAPPY' )

        volume_buffer = snappy.decompress(snappy_data)
        volume = np.frombuffer(volume_buffer, dtype=self.dtype).reshape(shape_zyx)
        return volume


    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = np.dtype(self.geometries[0]['channelType'].lower())
        return self._dtype


    @property
    def bounding_box(self):
        """
        The bounding box [start, stop] of the volume at scale 0.
        """
        return self.bounding_boxes[0] # Scale 0


    @property
    def bounding_boxes(self):
        """
        A list of bounding boxes (one per scale)
        """
        if self._bounding_boxes is None:
            self._bounding_boxes = list(map(self._extract_bounding_box, self.geometries))
        return self._bounding_boxes


    def __getstate__(self):
        """
        Pickle representation.
        """
        d = self.__dict__.copy()
        # Don't attempt to pickle the http connection, because
        # it would no longer be valid (authenticated) after it is unpickled.
        # Instead, set it to None so it will be lazily regenerated after unpickling.
        d['_http'] = None
        return d


    @property
    def http(self):
        """
        Returns an authenticated httplib2.Http object to use for all BrainMaps API requests.
        Memoized here instead of generated in the constructor,
        since we intentionally delete the _http member during pickling.
        """
        if self._http is not None:
            return self._http

        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if  not credentials_path:
            raise RuntimeError("To access BrainMaps volumes, you must define GOOGLE_APPLICATION_CREDENTIALS "
                               "in your environment, which must point to a google service account json credentials file.")

        scopes = ['https://www.googleapis.com/auth/brainmaps']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scopes)
        self._http = credentials.authorize(Http())
        return self._http


    @property
    def geometries(self):
        """
        The (memoized) geometry json for all scales.
        See get_geometry() for details.
        """
        if self._geometries is None:
            self._geometries = self.get_geometry()
            assert int(self._geometries[0]['channelCount']) == 1, \
                "Can't use this class on multi-channel volumes."
        return self._geometries


    def get_change_stacks(self):
        """
        Get the list of change_stacks 
        """
        msg_json = fetch_json(self.http, f'{BRAINMAPS_BASE_URL}/changes/{self.project}:{self.dataset}:{self.volume_id}/change_stacks')
        return msg_json['changeStackId']
        

    def get_geometry(self):
        """
        Returns a list of geometries (one per scale):
        
        [{
          'boundingBox': [{
            'corner': {},
            'size': {'x': '37911', 'y': '7731', 'z': '30613'}
          }],
          'channelCount': '1',
          'channelType': 'UINT64',
          'pixelSize': {'x': 8, 'y': 8, 'z': 8},
          'volumeSize': {'x': '37911', 'y': '7731', 'z': '30613'}
        },
        ...]

        (Notice that many of these numbers are strings, for some reason.) 
        """
        msg_json = fetch_json( self.http, f'{BRAINMAPS_BASE_URL}/volumes/{self.project}:{self.dataset}:{self.volume_id}')
        return msg_json['geometry']


    def _extract_bounding_box(self, geometry):
        """
        Return the bounding box [start, stop] in zyx order.
        """
        corner = geometry['boundingBox'][0]['corner']
        size = geometry['boundingBox'][0]['size']

        shape = [int(size[k]) for k in 'zyx']
        if not corner:
            offset = (0,)*len(size)
        else:
            offset = [int(corner[k]) for k in 'zyx']

        box = np.array((offset, offset))
        box[1] += shape
        return box


    def get_equivalence_group(self, segment_id):
        """
        Return the set of segments (supervoxels) that are in the same group as the given segment_id.
        """
        assert self.change_stack_id, "No equivalences: This volume has no change_stack_id"
        url = f'{BRAINMAPS_BASE_URL}/changes/{self.project}:{self.dataset}:{self.volume_id}/{self.change_stack_id}/equivalences:getgroups'

        # Technically we could give a list here, but we just ask for one at a time.
        equivalences = fetch_json(self.http, url, body={'segmentId': [segment_id]})
        
        # json data is
        # 
        # {
        #   'groups':
        #   [
        #     {
        #       'groupMembers':
        #       [
        #         '411313',
        #         '828042',
        #         '1239392',
        #         ...
        #       ]
        #     },
        #     ...
        #   ]
        # }
        
        # Convert from string to int
        siblings = set(map(int, equivalences["groups"][0]["groupMembers"]))
        assert segment_id in siblings, \
            f"Expected query segment ({segment_id}) to be in its own equivalence set: {siblings}"

        return siblings


    def get_all_equivalence_groups(self):
        """
        Like get_equivalences(), but returns results for all groups.
        
        Return a dictionary of { group_1 : [segment_1, segment_2, ...],
                                 group_2 : [segment_10, segment_11, ...],
                                 ... }
        """
        assert self.change_stack_id, "No equivalences: This volume has no change_stack_id"

        # There is no API for getting all groups, so we have to get the
        # full set of edges and run connected components ourselves.
        import networkx as nx
        
        all_edges = self.get_equivalence_edges()
        g = nx.Graph()
        g.add_edges_from(all_edges)
        
        groups = {}
        for segment_set in nx.connected_components(g):
            # According to Jeremy, group_id == the min segment_id of the group.
            groups[min(segment_set)] = list(sorted(segment_set))

        return groups


    def get_equivalence_edges(self, segment_id=None):
        """
        Get the merged edges for the group that owns the given segment as a numpy array [[s1, s2], [s1, s2], ...].
        If segment_id is None, return all equivalence edges in the entired volume.
        """
        assert self.change_stack_id, "No equivalences: This volume has no change_stack_id"

        url = f'{BRAINMAPS_BASE_URL}/changes/{self.project}:{self.dataset}:{self.volume_id}/{self.change_stack_id}/equivalences:list'
        if segment_id is None:
            edges_json = fetch_json(self.http, url, body={})
        else:
            edges_json = fetch_json(self.http, url, body={'segmentId': segment_id})

        # Data comes back in this format:
        # 
        # { 'edge': 
        #   [ 
        #     {'first': '411313', 'second': '1033743'},
        #     {'first': '411313', 'second': '1239364'},
        #     {'first': '411313', 'second': '1442313'},
        #     ...
        #   ]
        # }

        num_edges = len(edges_json['edge'])
        firsts = (edge['first'] for edge in edges_json['edge'])
        seconds = (edge['second'] for edge in edges_json['edge'])

        edges_flat = np.fromiter(chain(firsts, seconds), np.uint64, 2*num_edges)
        edges = edges_flat.reshape((2,-1)).transpose()
        return edges


def fetch_json(http, url, body=None):
    """
    Fetch JSON data from a BrainMaps API endpoint.
    
    Args:
        http: Authenticated httplib2.Http object.
        url: Full url to the endpoint.
        body: If the endpoint requires parameters in the body,
              give them here (as a dict). Forces method to be POST.
    
    Examples:
    
        projects = fetch_json(http, f'{BRAINMAPS_BASE_URL}/projects')
        volume_list = fetch_json(http, f'{BRAINMAPS_BASE_URL}/volumes')
    """
    if body is None:
        method = "GET"
    else:
        method = "POST"
        if not isinstance(body, (str, bytes)):
            body = json.dumps(body, cls=NumpyConvertingEncoder)

    response, content = http.request(url, method, body)
    if response['status'] != '200':
        raise RuntimeError(f"Bad response ({response['status']}):\n{content.decode('utf-8')}")

    return json.loads(content)


def fetch_subvol_data(http, project, dataset, volume_id, corner_xyz, size_xyz, scale, change_stack_id="", subvol_format='RAW_SNAPPY'):
    """
    Returns raw subvolume data (not decompressed).
    
    Clients should generally not call this function directly.
    Instead, use the BrainMapsVolume class.
    """
    url = f'{BRAINMAPS_BASE_URL}/volumes/{project}:{dataset}:{volume_id}/subvolume:binary'

    params = \
    {
        'geometry': {
            'corner': ','.join(str(x) for x in corner_xyz),
            'size': ','.join(str(x) for x in size_xyz),
            'scale': scale
        },
        'subvolumeFormat': subvol_format
    }

    if change_stack_id:
        params["changeSpec"] = { "changeStackId": change_stack_id }
    
    response, content = http.request(url, "POST", body=json.dumps(params).encode('utf-8'))
    if response['status'] != '200':
        raise RuntimeError(f"Bad response ({response['status']}):\n{content.decode('utf-8')}")
    return content


class NumpyConvertingEncoder(json.JSONEncoder):
    """
    Encoder that converts numpy arrays and scalars
    into their pure-python counterparts.
    
    (No attempt is made to preserve bit-width information.)
    
    Usage:
    
        >>> d = {"a": np.arange(3, dtype=np.uint32)}
        >>> json.dumps(d, cls=NumpyConvertingEncoder)
        '{"a": [0, 1, 2]}'
    """
    def default(self, o):
        if isinstance(o, (np.ndarray, np.number)):
            return o.tolist()
        return super().default(o)
