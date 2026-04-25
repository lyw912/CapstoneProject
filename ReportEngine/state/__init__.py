"""
Report Engine state management module.

Exports ReportState/ReportMetadata for sharing between Agent and Flask interface.
"""

from .state import ReportState, ReportMetadata

__all__ = ["ReportState", "ReportMetadata"]
