# THIS QUERY WILL JOIN THE APPLICATION SET TO THE HELD OUT H3 CELLS TO VALIDATE PERFORMANCE

INGENIOUS_HOT_SPRINGS = """
    with hot_springs as (
        select
              H3_LATLNG_TO_CELL_STRING( LATDEGREE, LONGDEGREE, 8) as H3_BLOCKS
        from DB_SCRATCH.RAW.INGENIOUS_SPRING_TEMPERATURE 
        where MEASUREDTEMP_C >= 40
        GROUP BY H3_BLOCKS
      )
    SELECT DISTINCT
        hot_springs.H3_BLOCKS,
        pfa_table.score as PRED,
        1 as LABEL
    FROM
        hot_springs
    INNER JOIN
        {APPLICATION_SET_TABLE} as pfa_table
    ON 
        hot_springs.H3_BLOCKS = pfa_table.H3_BLOCKS
"""
