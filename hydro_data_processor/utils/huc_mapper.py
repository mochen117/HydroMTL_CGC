"""
HUC Mapping utility for CAMELS data structure.
Maps basin_id to huc_02 and provides validation.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class HucMapper:
    """Manages basin_id -> huc_02 mapping for CAMELS data."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.basin_to_huc: Dict[str, str] = {}
        self.huc_to_basins: Dict[str, List[str]] = {}

        self._load_mapping()

    def _load_mapping(self):
        """Load huc mapping from camels_name.txt."""
        name_file = self.data_root / "camels" / "camels_us" / "camels_name.txt"

        if not name_file.exists():
            logger.warning(f"camels_name.txt not found at {name_file}")
            return

        try:
            df = pd.read_csv(name_file, delimiter=';')

            for _, row in df.iterrows():
                basin_id = str(row['gauge_id']).strip()
                huc_02 = str(row['huc_02']).strip().zfill(2)

                self.basin_to_huc[basin_id] = huc_02

                if huc_02 not in self.huc_to_basins:
                    self.huc_to_basins[huc_02] = []
                self.huc_to_basins[huc_02].append(basin_id)

            logger.info(
                f"Loaded HUC mapping: {len(self.basin_to_huc)} basins, "
                f"{len(self.huc_to_basins)} huc regions")

        except Exception as e:
            logger.error(f"Failed to load huc mapping from {name_file}: {e}")

    def get_huc(self, basin_id: str) -> Optional[str]:
        """Get huc_02 for a basin_id."""
        return self.basin_to_huc.get(basin_id)

    def get_basins_in_huc(self, huc_02: str) -> List[str]:
        """Get all basin_ids in a huc region."""
        return self.huc_to_basins.get(huc_02, [])

    def get_all_basins(self) -> List[str]:
        """Get all basin_ids."""
        return list(self.basin_to_huc.keys())

    def validate_basin(self, basin_id: str) -> bool:
        """Check if basin_id exists in mapping."""
        return basin_id in self.basin_to_huc

    def get_basin_name(self, basin_id: str) -> Optional[str]:
        """Get the name of a basin."""
        # This would require loading additional data from camels_name.txt
        return None
