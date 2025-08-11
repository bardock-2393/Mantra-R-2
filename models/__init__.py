"""
Models package for AI Video Detective
Contains model implementations and instances
"""

from .minicpm_v26_model import MiniCPMV26Model

# Create global model instances
minicpm_v26_model = MiniCPMV26Model()

__all__ = ['MiniCPMV26Model', 'minicpm_v26_model'] 