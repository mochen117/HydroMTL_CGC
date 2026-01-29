"""
Dataset builder for creating unified datasets
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build and manage datasets."""

    def __init__(self, output_dir: Path, output_format: str = "netcdf"):
        """
        Initialize dataset builder.

        Args:
            output_dir: Output directory
            output_format: Output file format
        """
        self.output_dir = output_dir
        self.output_format = output_format

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_basin_dataset(self, dataset: xr.Dataset, basin_id: str) -> bool:
        """
        Save basin dataset to file.

        Args:
            dataset: xarray Dataset
            basin_id: Basin ID

        Returns:
            True if successful
        """
        try:
            # Determine file extension
            if self.output_format == "netcdf":
                extension = ".nc"
            elif self.output_format == "hdf5":
                extension = ".h5"
            elif self.output_format == "parquet":
                extension = ".parquet"
            else:
                logger.error(
                    "Unsupported output format: %s",
                    self.output_format)
                return False

            # Build file path
            file_path = self.output_dir / f"{basin_id}{extension}"

            # Save based on format
            if self.output_format == "netcdf":
                encoding = {}
                for var in dataset.data_vars:
                    encoding[var] = {
                        'zlib': True,
                        'complevel': 4,
                        'dtype': 'float32'
                    }
                dataset.to_netcdf(file_path, encoding=encoding)

            elif self.output_format == "hdf5":
                dataset.to_netcdf(file_path, engine='h5netcdf')

            elif self.output_format == "parquet":
                df = dataset.to_dataframe()
                df.to_parquet(file_path)

            logger.debug(
                "Saved dataset for basin %s to %s",
                basin_id,
                file_path)
            return True

        except Exception as e:
            logger.error(
                "Failed to save dataset for basin %s: %s",
                basin_id,
                str(e))
            return False

    def combine_datasets(self, basin_ids: List[str]) -> Optional[xr.Dataset]:
        """
        Combine multiple basin datasets into one.

        Args:
            basin_ids: List of basin IDs

        Returns:
            Combined xarray Dataset or None
        """
        try:
            datasets = []

            for basin_id in basin_ids:
                # Try to load basin dataset
                basin_dataset = self._load_basin_dataset(basin_id)

                if basin_dataset is not None:
                    # Add basin dimension
                    basin_dataset = basin_dataset.expand_dims(
                        {'basin': [basin_id]})
                    datasets.append(basin_dataset)

            if not datasets:
                logger.warning("No datasets to combine")
                return None

            # Combine all datasets
            combined = xr.concat(datasets, dim='basin')

            # Add global attributes
            combined.attrs['title'] = 'CAMELS-US Extended Dataset (2001-2021)'
            combined.attrs['description'] = 'Multi-source hydrological dataset following paper methodology'
            combined.attrs['number_of_basins'] = len(datasets)
            combined.attrs['basin_ids'] = ','.join(basin_ids)
            combined.attrs['creation_date'] = pd.Timestamp.now().isoformat()

            logger.info(
                "Combined %d basins into single dataset",
                len(datasets))

            return combined

        except Exception as e:
            logger.error("Failed to combine datasets: %s", str(e))
            return None

    def save_combined_dataset(self, dataset: xr.Dataset,
                              filename: str = "combined_basins") -> bool:
        """
        Save combined dataset to file.

        Args:
            dataset: Combined xarray Dataset
            filename: Base filename (without extension)

        Returns:
            True if successful
        """
        try:
            # Determine file extension
            if self.output_format == "netcdf":
                file_path = self.output_dir / f"{filename}.nc"
                encoding = {}
                for var in dataset.data_vars:
                    encoding[var] = {
                        'zlib': True,
                        'complevel': 4,
                        'dtype': 'float32'
                    }
                dataset.to_netcdf(file_path, encoding=encoding)

            elif self.output_format == "hdf5":
                file_path = self.output_dir / f"{filename}.h5"
                dataset.to_netcdf(file_path, engine='h5netcdf')

            elif self.output_format == "parquet":
                file_path = self.output_dir / f"{filename}.parquet"
                df = dataset.to_dataframe()
                df.to_parquet(file_path)

            logger.info("Saved combined dataset to %s", file_path)
            logger.info("Dataset dimensions: %s", dataset.dims)

            return True

        except Exception as e:
            logger.error("Failed to save combined dataset: %s", str(e))
            return False

    def _load_basin_dataset(self, basin_id: str) -> Optional[xr.Dataset]:
        """
        Load basin dataset from file.

        Args:
            basin_id: Basin ID

        Returns:
            xarray Dataset or None
        """
        try:
            # Determine file extension
            if self.output_format == "netcdf":
                extension = ".nc"
            elif self.output_format == "hdf5":
                extension = ".h5"
            elif self.output_format == "parquet":
                # For parquet, we would need different handling
                return None
            else:
                return None

            file_path = self.output_dir / f"{basin_id}{extension}"

            if not file_path.exists():
                return None

            if self.output_format == "netcdf":
                return xr.open_dataset(file_path)
            elif self.output_format == "hdf5":
                return xr.open_dataset(file_path, engine='h5netcdf')

        except Exception as e:
            logger.warning(
                "Failed to load dataset for basin %s: %s",
                basin_id,
                str(e))
            return None
