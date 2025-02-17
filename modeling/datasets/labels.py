from modeling.datasets.base import BaseLabels
from modeling.datasets.build import LABEL_REGISTRY

@LABEL_REGISTRY.register()
class fishing(BaseLabels):
    table_name = "FISHING_RAW_DATA"
    index_column = "H3_BLOCKS"
    sql_code = """
    SELECT 
    fishing
    FROM FISHING_DB.FISHING_SCHEMA.FISHING_RAW_DATA
    """
