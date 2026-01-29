"""
File finder utility for CAMELS data structure.
Finds files based on basin_id and data type.
"""

from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CamelsFileFinder:
    """Finds files in CAMELS directory structure."""

    # Define base paths for different data types
    BASE_PATHS = {
        "camels_streamflow": [
            "camels/camels_us/camels_streamflow",
            "camels/camels_us/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/usgs_streamflow",
        ],
        "daymet_forcing": [
            "camels/camels_us/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet",
        ],
        "nldas_forcing": [
            "nldas4camels/basin_mean_forcing",
        ],
        "smap": [
            "smap4camels/SMAP_CAMELS",
        ],
        "et": []  # ET data location needs to be determined
    }

    # Define file patterns for different data types
    FILE_PATTERNS = {
        "camels_streamflow": [
            "{huc_02}/{basin_id}_streamflow_qc.txt",
        ],
        "daymet_forcing": [
            "{huc_02}/{basin_id}_lump_cida_forcing_leap.txt",
        ],
        "nldas_forcing": [
            "{huc_02}/{basin_id}_lump_nldas_forcing_leap.txt",
        ],
        "smap": [
            "{huc_02}/{basin_id}_lump_smap.txt",
        ],
        "et": []  # ET file pattern needs to be determined
    }

    def __init__(self, data_root: Path, huc_mapper):
        self.data_root = Path(data_root)
        self.huc_mapper = huc_mapper

    def find_file(self, basin_id: str, data_type: str) -> Optional[Path]:
        """
        Find a file for a specific basin and data type.

        Args:
            basin_id: 8-digit basin ID
            data_type: One of "camels_streamflow", "daymet_forcing",
                      "nldas_forcing", "smap", "et"

        Returns:
            Path to file if found, None otherwise
        """
        if data_type not in self.FILE_PATTERNS:
            logger.error(f"Unknown data type: {data_type}")
            return None

        # Get huc for this basin
        huc_02 = self.huc_mapper.get_huc(basin_id)

        if not huc_02:
            logger.warning(f"No huc_02 found for basin {basin_id}")
            return None

        # Try each base path
        for base_path in self.BASE_PATHS.get(data_type, []):
            base_dir = self.data_root / base_path

            if not base_dir.exists():
                continue

            # Try each file pattern
            for pattern in self.FILE_PATTERNS[data_type]:
                file_path_str = pattern.format(
                    huc_02=huc_02, basin_id=basin_id)
                file_path = base_dir / file_path_str

                if file_path.exists():
                    logger.debug(f"Found {data_type} file: {file_path}")
                    return file_path

        logger.warning(f"No {data_type} file found for basin {basin_id}")
        return None
