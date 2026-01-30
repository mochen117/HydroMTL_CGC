"""
Configuration settings for Hydro Data Processor.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""
    name: str
    data_source_path: Path
    file_pattern: str = ""
    subdirectory: str = ""
    delimiter: str = r'\s+'
    required: bool = True
    description: str = ""

    def __post_init__(self):
        """Initialize and resolve paths."""
        if isinstance(self.data_source_path, str):
            self.data_source_path = Path(self.data_source_path)

        try:
            if not self.data_source_path.is_absolute():
                self.data_source_path = self.data_source_path.resolve()
            else:
                self.data_source_path = self.data_source_path.resolve()
        except Exception as e:
            logger.debug(f"Could not resolve path for '{self.name}': {e}")

    def validate(self) -> bool:
        """Validate that the data source directory exists."""
        if self.required and not self.data_source_path.exists():
            logger.error(f"Required data source '{self.name}' not found at: {self.data_source_path}")
            return False
        elif not self.data_source_path.exists():
            logger.debug(f"Optional data source '{self.name}' not found at: {self.data_source_path}")
        else:
            logger.debug(f"Data source '{self.name}' validated: {self.data_source_path}")
        return True

    def get_file_path(self, gauge_id: str = "", huc2: str = "") -> Path:
        """Construct full file path with optional placeholders."""
        path = self.data_source_path

        if self.subdirectory:
            subdir = self.subdirectory
            if "{huc2}" in subdir and huc2:
                subdir = subdir.format(huc2=huc2)
            path = path / subdir

        if self.file_pattern:
            filename = self.file_pattern
            if "{gauge_id}" in filename and gauge_id:
                formatted_gauge_id = self._format_gauge_id_for_camels(gauge_id)
                filename = filename.format(gauge_id=formatted_gauge_id)
            if "{huc2}" in filename and huc2:
                filename = filename.format(huc2=huc2)
            path = path / filename

        return path

    def _format_gauge_id_for_camels(self, gauge_id: str) -> str:
        """Format gauge_id to 8-digit string for CAMELS files."""
        try:
            clean_id = ''.join(filter(str.isdigit, str(gauge_id)))

            if len(clean_id) == 7:
                return '0' + clean_id
            elif len(clean_id) == 8:
                return clean_id
            else:
                return clean_id.zfill(8)
        except BaseException:
            return str(gauge_id).zfill(8)


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    start_date: str = "2001-01-01"
    end_date: str = "2021-09-30"
    min_streamflow_coverage: float = 0.95
    output_format: str = "netcdf"
    overwrite_existing: bool = False
    chunk_size: int = 100
    parallel_processing: bool = True
    max_workers: int = None

    def validate(self) -> List[str]:
        """Validate processing parameters."""
        errors = []

        try:
            from datetime import datetime
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError:
            errors.append("Invalid date format. Use YYYY-MM-DD")

        if not 0 <= self.min_streamflow_coverage <= 1:
            errors.append("min_streamflow_coverage must be between 0 and 1")

        valid_formats = ["netcdf", "parquet", "csv"]
        if self.output_format not in valid_formats:
            errors.append(f"Invalid output format. Must be one of: {valid_formats}")

        return errors


@dataclass
class ProjectConfig:
    """Main project configuration."""

    project_name: str = "HydroDataProcessor"
    version: str = "1.0.0"
    data_root: Path = field(default_factory=lambda:
                            Path(os.environ.get("HYDRO_DATA_ROOT", "./data")))
    max_basins: int = 10
    selected_basins: List[str] = field(default_factory=list)
    data_sources: Dict[str, DataSourceConfig] = field(default_factory=dict)
    processing_config: ProcessingConfig = field(
        default_factory=ProcessingConfig)
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def __post_init__(self):
        """Initialize and validate configuration."""
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        self.data_root = self.data_root.resolve()

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir = self.output_dir.resolve()

        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
        if self.log_file:
            self.log_file = self.log_file.resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_sources:
            self._init_default_data_sources()

        for name, source in self.data_sources.items():
            if not hasattr(source, 'name') or not source.name:
                source.name = name

        logger.debug(f"Resolved data_root: {self.data_root}")
        logger.debug(f"Resolved output_dir: {self.output_dir}")

    def _init_default_data_sources(self):
        """Initialize default data source configurations."""
        # Use the data_root directly for flexibility
        data_root = self.data_root

        self.data_sources = {
            "attributes": DataSourceConfig(
                name="basin_attributes",
                data_source_path=data_root / "camels" / "camels_us",
                file_pattern="camels_name.txt",
                delimiter=";",
                description="CAMELS basin attributes and HUC2 mapping"),
            
            "camels_streamflow": DataSourceConfig(
                name="camels_streamflow",
                data_source_path=data_root / "camels" / "camels_us" / 
                                  "basin_timeseries_v1p2_metForcing_obsFlow" /
                                  "basin_dataset_public_v1p2" / "camels_streamflow",
                file_pattern="{gauge_id}_streamflow_qc.txt",
                subdirectory="{huc2}",
                delimiter=r'\s+',  # Whitespace delimiter for CAMELS data
                description="CAMELS USGS streamflow data (1980-2014)"),
            
            "usgs_streamflow": DataSourceConfig(
                name="usgs_streamflow",
                data_source_path=data_root / "camels" / "camels_us" / "usgs_streamflow",
                file_pattern="{gauge_id}_streamflow_qc.txt",
                subdirectory="{huc2}",
                delimiter=',',  # Comma delimiter for USGS data
                required=False,  # Optional for backward compatibility
                description="USGS streamflow data (2015-2021)"),
            
            "nldas_forcing": DataSourceConfig(
                name="nldas_forcing",
                data_source_path=data_root / "nldas4camels",
                file_pattern="{gauge_id}_lump_nldas_forcing_leap.txt",
                subdirectory="basin_mean_forcing/{huc2}",
                delimiter=r'\s+',
                description="NLDAS meteorological forcing data"),
            
            "et_data": DataSourceConfig(
                name="modis_et",
                data_source_path=data_root / "modiset4camels",
                file_pattern="{gauge_id}_lump_modis16a2v006_et.txt",
                subdirectory="basin_mean_forcing/MOD16A2_006_CAMELS/{huc2}",
                delimiter=r'\s+',
                required=False,
                description="MODIS ET data (optional)"),
            
            "smap_data": DataSourceConfig(
                name="smap_soil_moisture",
                data_source_path=data_root / "smap4camels",
                file_pattern="{gauge_id}_lump_nasa_usda_smap.txt",
                subdirectory="NASA_USDA_SMAP_CAMELS/{huc2}",
                delimiter=r'\s+',
                required=False,
                description="SMAP soil moisture data (optional)")
        }

        logger.debug(f"Initialized {len(self.data_sources)} data sources")

    def validate(self) -> Dict[str, List[str]]:
        """Validate entire configuration."""
        errors = {
            "data_sources": [],
            "processing": [],
            "paths": []
        }

        if not self.data_root.exists():
            errors["paths"].append(
                f"Data root directory does not exist: {self.data_root}")
            self.data_root.mkdir(parents=True, exist_ok=True)

        for name, source in self.data_sources.items():
            if not source.validate():
                if source.required:
                    errors["data_sources"].append(
                        f"Required data source '{name}' validation failed")

        processing_errors = self.processing_config.validate()
        if processing_errors:
            errors["processing"].extend(processing_errors)

        try:
            test_file = self.output_dir / ".config_test"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            errors["paths"].append(
                f"Cannot write to output directory {self.output_dir}: {e}")

        return {k: v for k, v in errors.items() if v}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProjectConfig':
        """Create ProjectConfig from a dictionary."""
        project_name = config_dict.get('project_name', 'HydroDataProcessor')
        version = config_dict.get('version', '1.0.0')
        
        data_root_str = config_dict.get('data_root')
        if data_root_str:
            data_root_str = os.path.expanduser(os.path.expandvars(data_root_str))
            data_root = Path(data_root_str)
        else:
            data_root = Path(os.environ.get("HYDRO_DATA_ROOT", "./data"))
        
        data_root = data_root.resolve()
        
        max_basins = config_dict.get('max_basins', 10)
        selected_basins = config_dict.get('selected_basins', [])
        
        # Processing config
        proc_config_dict = config_dict.get('processing_config', {})
        processing_config = ProcessingConfig(
            start_date=proc_config_dict.get('start_date', '2001-01-01'),
            end_date=proc_config_dict.get('end_date', '2021-09-30'),
            min_streamflow_coverage=proc_config_dict.get('min_streamflow_coverage', 0.95),
            output_format=proc_config_dict.get('output_format', 'netcdf'),
            overwrite_existing=proc_config_dict.get('overwrite_existing', False),
            chunk_size=proc_config_dict.get('chunk_size', 100),
            parallel_processing=proc_config_dict.get('parallel_processing', True),
            max_workers=proc_config_dict.get('max_workers')
        )
        
        # Output directory
        output_dir_str = config_dict.get('output_dir', './output')
        output_dir = Path(os.path.expanduser(os.path.expandvars(output_dir_str))).resolve()
        
        # Create config instance
        config = cls(
            project_name=project_name,
            version=version,
            data_root=data_root,
            max_basins=max_basins,
            selected_basins=selected_basins,
            processing_config=processing_config,
            output_dir=output_dir,
            log_level=config_dict.get('log_level', 'INFO'),
            log_file=config_dict.get('log_file')
        )
        
        # Validate
        errors = config.validate()
        if errors:
            logger.warning(f"Configuration validation warnings: {errors}")
        
        return config


def get_default_config() -> ProjectConfig:
    config = ProjectConfig()

    env_data_root = os.environ.get("HYDRO_DATA_ROOT")
    if env_data_root:
        config.data_root = Path(env_data_root).resolve()
        logger.info(f"Using data root from environment: {config.data_root}")

    return config


config = get_default_config()