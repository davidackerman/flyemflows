"""Defines class for compressing/decompressing serialized RDD data.

There are numerous challenges with using python for large datasets
in Spark.  In Spark, RDDs frequently need to be serialized/deserialized
as data is shuffled or cached.  However, in python, all cached
data is serialize (unlike in Java).  Futhermore, the flexibility
in python often results in inefficiently stored data.

To help reduce the size of the shuffled data, we look to compression.
While limited compression is available in pyspark, the following
is a very light-weight compression, lz4, that appears to negligibly impact
runtime while leading to good compression (for instance sparse
numpy array volumes can be shrunk by over 10x is a fraction of the
time it takes to cPickle the object).  While extensive performance
tests have not been performed, it seems likely to not be any bottleneck
compared to the pickler's performance.

"""

from pyspark.serializers import FramedSerializer, PickleSerializer 
import lz4

class CompressedSerializerLZ4(FramedSerializer):
    """ Compress/decompress already serialized data using fast lz4.

        Note: extensive performance testing is still
        necessary.  It might be a candidate for inclusion
        within the pyspark distribution.

    """

    def __init__(self, serializer=PickleSerializer()):
        FramedSerializer.__init__(self)
        assert isinstance(serializer, FramedSerializer), "serializer must be a FramedSerializer"
        self.serializer = serializer

    def dumps(self, obj):
        return lz4.dumps(self.serializer.dumps(obj))

    def loads(self, obj):
        return self.serializer.loads(lz4.loads(obj))

    def __repr__(self):
        return "CompressedSerializerLZ4(%s)" % self.serializer
