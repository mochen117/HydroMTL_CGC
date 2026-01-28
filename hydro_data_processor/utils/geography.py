"""
Geography utilities for Hydro Data Processor
"""

import re
from typing import List, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


def extract_huc2(basin_id: str) -> str:
    """
    Extract HUC2 code from basin ID.
    
    Args:
        basin_id: Basin ID
        
    Returns:
        2-character HUC2 code
    """
    # Ensure basin_id is string
    basin_id = str(basin_id)
    
    # Pad with zeros to ensure 8 characters
    basin_id = basin_id.zfill(8)
    
    # Extract first 2 characters
    return basin_id[:2]


def extract_huc4(basin_id: str) -> str:
    """
    Extract HUC4 code from basin ID.
    
    Args:
        basin_id: Basin ID
        
    Returns:
        4-character HUC4 code
    """
    basin_id = str(basin_id).zfill(8)
    return basin_id[:4]


def extract_huc6(basin_id: str) -> str:
    """
    Extract HUC6 code from basin ID.
    
    Args:
        basin_id: Basin ID
        
    Returns:
        6-character HUC6 code
    """
    basin_id = str(basin_id).zfill(8)
    return basin_id[:6]


def group_basins_by_huc(basin_ids: List[str], 
                       huc_level: int = 2) -> Dict[str, List[str]]:
    """
    Group basin IDs by HUC code.
    
    Args:
        basin_ids: List of basin IDs
        huc_level: HUC level (2, 4, or 6)
        
    Returns:
        Dictionary mapping HUC codes to lists of basin IDs
    """
    groups = {}
    
    for basin_id in basin_ids:
        if huc_level == 2:
            huc_code = extract_huc2(basin_id)
        elif huc_level == 4:
            huc_code = extract_huc4(basin_id)
        elif huc_level == 6:
            huc_code = extract_huc6(basin_id)
        else:
            raise ValueError(f"Invalid HUC level: {huc_level}")
        
        if huc_code not in groups:
            groups[huc_code] = []
        groups[huc_code].append(basin_id)
    
    return groups


def validate_basin_id(basin_id: str) -> bool:
    """
    Validate basin ID format.
    
    Args:
        basin_id: Basin ID to validate
        
    Returns:
        True if valid format
    """
    if not isinstance(basin_id, str):
        return False
    
    # Check length
    if len(basin_id) != 8:
        return False
    
    # Check if all characters are digits
    if not basin_id.isdigit():
        return False
    
    return True


def get_huc_region_name(huc2: str) -> Optional[str]:
    """
    Get region name for HUC2 code.
    
    Args:
        huc2: 2-character HUC2 code
        
    Returns:
        Region name or None if not found
    """
    huc_regions = {
        '01': 'New England',
        '02': 'Mid Atlantic',
        '03': 'South Atlantic-Gulf',
        '04': 'Great Lakes',
        '05': 'Ohio',
        '06': 'Tennessee',
        '07': 'Upper Mississippi',
        '08': 'Lower Mississippi',
        '09': 'Souris-Red-Rainy',
        '10': 'Missouri',
        '11': 'Arkansas-White-Red',
        '12': 'Texas-Gulf',
        '13': 'Rio Grande',
        '14': 'Upper Colorado',
        '15': 'Lower Colorado',
        '16': 'Great Basin',
        '17': 'Pacific Northwest',
        '18': 'California'
    }
    
    return huc_regions.get(huc2)