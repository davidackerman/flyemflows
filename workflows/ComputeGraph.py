import httplib

from pydvid.labelgraph import labelgraph
from pydvid.errors import DvidHttpError

from recospark.reconutils import SimpleGraph
from recospark.reconutils.dvidworkflow import DVIDWorkflow

class ComputeGraph(DVIDWorkflow):
    # schema for building graph
    Schema = """
{ "$schema": "http://json-schema.org/schema#",
  "title": "Tool to Create DVID blocks from image slices",
  "type": "object",
  "properties": {
    "dvid-server": { 
      "description": "location of DVID server",
      "type": "string" 
    },
    "uuid": { "type": "string" },
    "label-name": { 
      "description": "DVID data instance pointing to label blocks",
      "type": "string" 
    },
    "roi": { 
      "description": "name of DVID ROI for given label-name",
      "type": "string" 
    },
    "graph-name": { 
      "description": "destination name of DVID labelgraph",
      "type": "string" 
    },
    "graph-builder-exe": { 
      "description": "name of executable that will build graph",
      "type": "string" 
    },
    "chunk-size": {
      "description": "size of chunks to be processed",
      "type": "integer",
      "default": 256
    }
  },
  "required" : ["dvid-server", "uuid", "label-name", "roi", "graph-name"]
}
    """

    chunksize = 256

    def __init__(self, config_filename):
        super(ComputeGraph, self).__init__(config_filename, self.Schema, "Compute Graph")

    # build graph by dividing into mostly disjoint chunks and computing the number
    # of voxels and overlap between them
    def execute(self):
        from pyspark import SparkContext
        from pyspark import StorageLevel

        if "chunk-size" in self.config_data:
            self.chunksize = self.config_data["chunk-size"]

        #  grab ROI
        distrois = self.sparkdvid_context.parallelize_roi(self.config_data["roi"],
                self.chunksize)

        # map ROI to label volume (1 pixel overlap)
        label_chunks = self.sparkdvid_context.map_labels64(distrois, self.config_data["label-name"], 1)

        # map labels to graph data -- external program (eventually convert neuroproof metrics and graph to a python library) ?!
        sg = SimpleGraph.SimpleGraph(self.config_data) 

        # extract graph
        graph_elements = label_chunks.flatMap(sg.build_graph) 

        # group data for vertices and edges
        graph_elements_red = graph_elements.reduceByKey(lambda a, b: a + b) 
        graph_elements_red.persist(StorageLevel.MEMORY_ONLY) # ??
        graph_vertices = graph_elements_red.filter(sg.is_vertex)
        graph_edges = graph_elements_red.filter(sg.is_edge)

        # create graph
        conn = httplib.HTTPConnection(self.config_data["dvid-server"])
        try:
            labelgraph.create_new(conn, self.config_data["uuid"], self.config_data["graph-name"])
        except DvidHttpError:
            pass

        # dump graph -- should this be wrapped through utils or through sparkdvid ??
        # will this result in too many request (should they be accumulated) ??
        # currently looking at one partitioning at a time to try to group requests
        self.sparkdvid_context.foreachPartition_graph_elements(graph_vertices,
                self.config_data["graph-name"])
        self.sparkdvid_context.foreachPartition_graph_elements(graph_edges,
                self.config_data["graph-name"])

        if "debug" in self.config_data and self.config_data["debug"]:
            num_elements = graph_elements.count()
            print "DEBUG: ", num_elements

        graph_elements_red.unpersist()

    @staticmethod
    def dumpschema():
        return ComputeGraph.Schema
