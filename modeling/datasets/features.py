import logging

from modeling.datasets.base import BaseFeatures
from modeling.datasets.build import FEATURES_REGISTRY

logger = logging.getLogger(__name__)


@FEATURES_REGISTRY.register()
class Ingenious_Outline_H3Grid(BaseFeatures):
    table_name = "INGENIOUS_H3GRID_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    # This is a shapefile we convert into H3
    sql_code = """
    SELECT 
    H3_BLOCKS
    FROM ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.INGENIOUS_GRID_EXTENT
    """


@FEATURES_REGISTRY.register()
class Ingenious_Depth_to_Basement(BaseFeatures):
    table_name = "DEPTH_TO_BASEMENT_GRID_INGENIOUS_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    # This is a raster we sample onto h3 directly
    sql_code = """
         SELECT
            H3_BLOCKS 
          , D2B_METERS
           FROM      
           ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.DEPTH_TO_BASEMENT_GRID_INGENIOUS
    """


@FEATURES_REGISTRY.register()
class Ingenious_CBA_HGM_Gravity(BaseFeatures):
    table_name = "GRAVITY_HGM_GRID_INGENIOUS_RESOLUTION_8_"
    index_column = "H3_BLOCKS"
    # This is a raster we sample onto h3 directly
    sql_code = """
      SELECT
      DISTINCT
        H3_BLOCKS
      , GRAVITY_HGM
      FROM 
      ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.GRAVITY_HGM_GRID_INGENIOUS
    """


@FEATURES_REGISTRY.register()
class Ingenious_Earthquake_Density_n100a15(BaseFeatures):
    table_name = "EARTHQUAKE_DENSITY_N100A15_GRID_INGENIOUS_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    # This is a raster we sample onto h3 directly
    sql_code = """
      SELECT
      DISTINCT
        H3_BLOCKS
      , EQ_DENSITY_N100A151
        FROM 
    ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.EARTHQUAKE_DENSITY_N100A15_GRID_INGENIOUS
      """


@FEATURES_REGISTRY.register()
class Ingenious_MT_Conductance_Layers(BaseFeatures):
    table_name = "CONDUCTANCE_MT_GRIDS_INGENIOUS_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    # This is a group of rasters we sample onto h3 directly
    sql_code = """
      SELECT
        H3_BLOCKS
      , CONDUCTANCE_LOW_CRUST 
      , CONDUCTANCE_SURFACE
      , CONDUCTANCE_MID_CRUST
      , CONDUCTANCE_MANTLE
        FROM
      ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.CONDUCTANCE_GRIDS_INGENIOUS
      """


@FEATURES_REGISTRY.register()
class Ingenious_Geodetic_Layers(BaseFeatures):
    table_name = "GEODETIC_GRIDS_INGENIOUS_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    # This is a raster we sample onto h3 directly
    sql_code = """   
    SELECT
        H3_BLOCKS,
        GEODETIC_2ND_INV,
        GEODETIC_DILATION_RATE,
        GEODETIC_SHEAR_RATE       
    FROM
        ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.GEODETIC_GRIDS_INGENIOUS
     """


@FEATURES_REGISTRY.register()
class Ingenious_Heatflow(BaseFeatures):
    table_name = "HEATFLOW_GRID_INGENIOUS_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    # This is a raster we sample onto h3 directly
    sql_code = """          
    SELECT
        H3_BLOCKS
        , HEATFLOW
        FROM 
        ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.HEATFLOW_GRID_INGENIOUS
    """


@FEATURES_REGISTRY.register()
class Ingenious_RTP_HGM_Magnetic(BaseFeatures):
    table_name = "MAGNETIC_HGM_GRID_INGENIOUS_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    # This is a raster we sample onto h3 directly
    sql_code = """
    SELECT
        H3_BLOCKS
        , MAGNETIC_HGM
        FROM 
        ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.MAGNETIC_RTP_HGM_GRID_INGENIOUS 
    """


@FEATURES_REGISTRY.register()
class Ingenious_Quaternary_Volcanics_Distance(BaseFeatures):
    table_name = "QUATERNARY_VOLCANICS_INGENIOUS_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    sql_code = """
    /* 
    GENERATE FEATURES FROM FELSIC, MAFIC, AND INTERMEDIATE CRATER (GB VENT) DATA
    COLUMNS:
        H3_BLOCKS : STR
            H3 IDS AT RESOLUTION 8
        FELSIC_CRATER_DIST : FLOAT
            MINIMUM DISTANCE IN KM TO THE NEAREST FELSIC CRATER (VENT) CELL
        MAFIC_CRATER_DIST : FLOAT
            MINIMUM DISTANCE IN KM TO THE NEAREST MAFIC CRATER (VENT) CELL
        INTERMEDIATE_CRATER_DIST : FLOAT
            MINIMUM DISTANCE IN KM TO THE NEAREST INTERMEDIATE CRATER (VENT) CELL
    */

    WITH ALL_GB_VENTS AS (
        SELECT DISTINCT
            ROCK_COMP,
            H3_LATLNG_TO_CELL_STRING(LAT, LONG, 8) AS H3_BLOCKS
        FROM ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.GB_VENTS_INGENIOUS
        WHERE ROCK_COMP IN ('felsic', 'mafic', 'intermediate')
    ),
    COUNT_CRATERS AS (
        SELECT DISTINCT
            H3_BLOCKS,
            ROCK_COMP,
            H3_GRID_DISK(H3_BLOCKS, 20) AS NEIGHBORS
        FROM ALL_GB_VENTS
        GROUP BY H3_BLOCKS, ROCK_COMP
    ),
    GRID AS (
        SELECT
            P.H3_BLOCKS,
            P.ROCK_COMP,
            TO_VARCHAR(F.VALUE) AS BLOCK_EXPANDED
        FROM COUNT_CRATERS AS P,
            LATERAL FLATTEN(INPUT => P.NEIGHBORS) F
    ),
    NEAREST_CRATER_DISTANCE AS (
        SELECT
            BLOCK_EXPANDED AS H3_BLOCKS,
            MIN(H3_GRID_DISTANCE(BLOCK_EXPANDED, H3_BLOCKS)) AS CRATER_DIST_KM
        FROM GRID
        GROUP BY BLOCK_EXPANDED
    )
    SELECT *
    FROM NEAREST_CRATER_DISTANCE
    """


@FEATURES_REGISTRY.register()
class Ingenious_Fault_Layers(BaseFeatures):
    table_name = "_FAULT_LAYERS_INGENIOUS_RESOLUTION_8"
    index_column = "H3_BLOCKS"
    # This is a line polygon. We convert to h3 then apply a distance transformation
    sql_code = """  
  WITH fault_res8 AS (
    -- Generate resolution 8 H3 cells covering the polygon
    SELECT 
        H3_COVERAGE_STRINGS(
            TO_GEOGRAPHY(ST_SETSRID(GEOMETRY, 4326)),
            8
        ) AS set_of_h3_res8,
        SLIPRTNUM,
        RECNUM
    FROM ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.FAULT_GEOMETRY_INGENIOUS
),
flattened_fault_res8 AS (
    -- Flatten the resolution 8 H3 cells into individual rows
    SELECT 
        VALUE AS H3_BLOCK_RES8,
        SLIPRTNUM,
        RECNUM
    FROM fault_res8,
    LATERAL FLATTEN(input => set_of_h3_res8)
),
neighbors AS (
    -- Generate neighbors within 5 H3 cells for each H3 block
    SELECT 
        H3_BLOCK_RES8 AS CENTER_H3_BLOCK,
        SLIPRTNUM,
        RECNUM,
        H3_GRID_DISK(H3_BLOCK_RES8, 10) AS neighbors
    FROM flattened_fault_res8
),
flattened_neighbors AS (
    -- Flatten neighbors into individual rows
    SELECT 
        TO_VARCHAR(f.VALUE) AS NEIGHBOR_H3_BLOCK,
        n.CENTER_H3_BLOCK,
        n.SLIPRTNUM,
        n.RECNUM
    FROM neighbors n,
    LATERAL FLATTEN(input => n.neighbors) f
),
aggregated_data AS (
    -- Aggregate MAX(SLIPRTNUM) and MIN(RECNUM) for each neighborhood cell
    SELECT 
        NEIGHBOR_H3_BLOCK AS H3_BLOCKS,
        MAX(SLIPRTNUM) AS MAX_SLIPRTNUM,
        MIN(RECNUM) AS MIN_RECNUM
    FROM flattened_neighbors
    GROUP BY NEIGHBOR_H3_BLOCK
)
SELECT DISTINCT 
    H3_BLOCKS,
    MAX_SLIPRTNUM,
    MIN_RECNUM
FROM aggregated_data
ORDER BY H3_BLOCKS
     """
