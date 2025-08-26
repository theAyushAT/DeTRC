from .version import __version__

# Import models to register them
from .model import DeTRC
from .repcount_dataset import RepCountDataset

__all__ = ['__version__', 'DeTRC', 'RepCountDataset']
