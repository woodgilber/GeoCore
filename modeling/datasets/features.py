import logging

from modeling.datasets.base import BaseFeatures
from modeling.datasets.build import FEATURES_REGISTRY

logger = logging.getLogger(__name__)


@FEATURES_REGISTRY.register()
class speed(BaseFeatures):
    table_name = "FISHING_RAW_DATA"
    # This is a shapefile we convert into H3
    sql_code = """
    SELECT 
    speed
    FROM FISHING_DB.FISHING_SCHEMA.FISHING_RAW_DATA
    """


@FEATURES_REGISTRY.register()
class heading(BaseFeatures):
    table_name = "FISHING_RAW_DATA"
    # This is a raster we sample onto h3 directly
    sql_code = """
         SELECT
            heading
           FROM      
           FISHING_DB.FISHING_SCHEMA.FISHING_RAW_DATA
    """


@FEATURES_REGISTRY.register()
class latitude(BaseFeatures):
    table_name = "FISHING_RAW_DATA"
    # This is a raster we sample onto h3 directly
    sql_code = """
      SELECT
      latitude
      FROM 
      FISHING_DB.FISHING_SCHEMA.FISHING_RAW_DATA
    """


@FEATURES_REGISTRY.register()
class longitude(BaseFeatures):
    table_name = "FISHING_RAW_DATA"
    # This is a raster we sample onto h3 directly
    sql_code = """
      SELECT
      longitude
        FROM 
    FISHING_DB.FISHING_SCHEMA.FISHING_RAW_DATA
      """


@FEATURES_REGISTRY.register()
class distance_from_coast(BaseFeatures):
    table_name = "FISHING_RAW_DATA"
    # This is a group of rasters we sample onto h3 directly
    sql_code = """
      SELECT
        distance_from_coast
        FROM
      FISHING_DB.FISHING_SCHEMA.FISHING_RAW_DATA
      """
