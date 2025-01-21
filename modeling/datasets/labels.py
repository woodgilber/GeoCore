from modeling.datasets.base import BaseLabels
from modeling.datasets.build import LABEL_REGISTRY


@LABEL_REGISTRY.register()
class Ingenious_Wells_TempC_Labels(BaseLabels):
    table_name = "INGENIOUS_LABELS"
    index_column = "H3_BLOCKS"
    sql_asis = True
    sql_code = """
 WITH ingenious_well_tempC_labels AS (
        SELECT 
            H3_LATLNG_TO_CELL_STRING(LATDEGREE,LONGDEGREE, 8) AS H3_BLOCKS,
            MAXMEASUREDTEMP_C ,
            DRILLERTOTALDEPTH_M,
            LONGDEGREE, LATDEGREE,
        CASE
            WHEN THERMALCLASS IN ('Cold') AND (DRILLERTOTALDEPTH_M >= {negative_well_depth_threshold} OR TRUEVERTICALDEPTH_M >= {negative_well_depth_threshold}) THEN 0
            WHEN THERMALCLASS IN ('Hot', 'Warm') AND (DRILLERTOTALDEPTH_M >= {positive_well_depth_threshold} OR TRUEVERTICALDEPTH_M >= {positive_well_depth_threshold}) THEN 1
            ELSE NULL
        END AS LABEL,
            0 AS WEIGHT
        FROM ZANSKAR_SNOWFLAKE_MARKETPLACE.INGENIOUS.INGENIOUS_WELL_FEATURES
        )
    SELECT 
      H3_BLOCKS
      , LABEL
      , WEIGHT
      , 'ingenious_well_tempC' as TYPE
    FROM
      ingenious_well_tempC_labels
      where LABEL is not null 
    """
