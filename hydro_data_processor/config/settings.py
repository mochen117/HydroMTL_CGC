"""
Configuration settings for Hydro Data Processor.
All paths are defined here and referenced throughout the codebase.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""
    data_source_path: Path
    file_pattern: str = ""
    subdirectory: str = ""
    delimiter: str = r'\s+'  # Default for CAMELS whitespace-separated files
    
    def __post_init__(self):
        if isinstance(self.data_source_path, str):
            self.data_source_path = Path(self.data_source_path)


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    start_date: str = "2001-01-01"
    end_date: str = "2021-09-30"
    min_streamflow_coverage: float = 0.95  # 5% missing max
    output_format: str = "netcdf"
    overwrite_existing: bool = False


@dataclass
class ProjectConfig:
    """Main project configuration."""
    
    # Root data directory
    data_root: Path
    
    # Maximum number of basins to process
    max_basins: int = 10
    
    # Selected basins (if empty, process all)
    selected_basins: List[str] = field(default_factory=list)
    
    # Data source configurations
    data_sources: Dict[str, DataSourceConfig] = field(default_factory=dict)
    
    # Processing configuration
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    
    def __post_init__(self):
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Initialize default data sources if not provided
        if not self.data_sources:
            self._init_default_data_sources()
    
    def _init_default_data_sources(self):
        """Initialize default data source configurations."""
        self.data_sources = {
            "attributes": DataSourceConfig(
                data_source_path=self.data_root / "camels" / "camels_us",
                file_pattern="",
                subdirectory=""
            ),
            "camels_streamflow": DataSourceConfig(
                data_source_path=self.data_root / "camels" / "camels_us" / 
                               "basin_timeseries_v1p2_metForcing_obsFlow" / 
                               "basin_dataset_public_v1p2" / "usgs_streamflow",
                file_pattern="{basin_id}_streamflow_qc.txt",
                subdirectory="{huc2}"
            ),
            "usgs_streamflow": DataSourceConfig(
                data_source_path=self.data_root / "camels" / "camels_us" / "camels_streamflow",
                file_pattern="{basin_id}_streamflow_qc.txt",
                subdirectory="{huc2}"
            ),
            "nldas_forcing": DataSourceConfig(
                data_source_path=self.data_root / "nldas4camels",
                file_pattern="{basin_id}_lump_nldas_forcing_leap.txt",
                subdirectory="basin_mean_forcing/{huc2}"
            ),
            "et_data": DataSourceConfig(
                data_source_path=self.data_root / "modiset4camels",
                file_pattern="{basin_id}_lump_modis16a2v006_et.txt",
                subdirectory="basin_mean_forcing/MOD16A2_006_CAMELS/{huc2}"
            ),
            "smap_data": DataSourceConfig(
                data_source_path=self.data_root / "smap4camels",
                file_pattern="{basin_id}_lump_nasa_usda_smap.txt",
                subdirectory="NASA_USDA_SMAP_CAMELS/{huc2}"
            )
        }
    
    def get_selected_basins(self) -> List[str]:
        """Get list of selected basins."""
        return self.selected_basins
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProjectConfig':
        """Create config from dictionary."""
        # Extract base config
        data_root = Path(config_dict.get('data_root', '.'))
        max_basins = config_dict.get('max_basins', 10)
        selected_basins = config_dict.get('selected_basins', [])
        
        # Extract processing config
        proc_config_dict = config_dict.get('processing_config', {})
        processing_config = ProcessingConfig(
            start_date=proc_config_dict.get('start_date', '2001-01-01'),
            end_date=proc_config_dict.get('end_date', '2021-09-30'),
            min_streamflow_coverage=proc_config_dict.get('min_streamflow_coverage', 0.95),
            output_format=proc_config_dict.get('output_format', 'netcdf'),
            overwrite_existing=proc_config_dict.get('overwrite_existing', False)
        )
        
        # Build data sources
        data_sources = {}
        for source_name, source_config in config_dict.get('data_sources', {}).items():
            data_sources[source_name] = DataSourceConfig(
                data_source_path=Path(source_config.get('data_source_path', '')),
                file_pattern=source_config.get('file_pattern', ''),
                subdirectory=source_config.get('subdirectory', ''),
                delimiter=source_config.get('delimiter', r'\s+')
            )
        
        return cls(
            data_root=data_root,
            max_basins=max_basins,
            selected_basins=selected_basins,
            data_sources=data_sources,
            processing_config=processing_config,
            output_dir=Path(config_dict.get('output_dir', './output'))
        )


# Default configuration
config = ProjectConfig(
    data_root=Path("/home/mochen/hydro_data"),
    max_basins=10,
    selected_basins=[]
)