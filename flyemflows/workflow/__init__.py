from .base.workflow import Workflow
from .connectedcomponents import ConnectedComponents
from .copygrayscale import CopyGrayscale
from .copysegmentation import CopySegmentation
from .decimatemeshes import DecimateMeshes
from .findadjacencies import FindAdjacencies
from .samplepoints import SamplePoints
from .sparsemeshes import SparseMeshes
from .stitchedmeshes import StitchedMeshes


BUILTIN_WORKFLOWS = [
    Workflow, # Base class, used for unit testing only
    ConnectedComponents,
    CopyGrayscale,
    CopySegmentation,
    DecimateMeshes,
    FindAdjacencies,
    SamplePoints,
    SparseMeshes,
    StitchedMeshes,
]
