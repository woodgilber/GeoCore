import logging
import os
import re
from abc import ABC, abstractmethod

import h3

logger = logging.getLogger(__name__)


class BadTable(Exception):
    """
    Custom exception for bad table
    """

    def __init__(self, n_entries, n_distincts):
        self.msg = f"""Number of entries {n_entries} in the table is not equal to the number of distinct indices \
            {n_distincts}. Exiting!"""
        return

    def __str__(self):
        return repr(self.msg)


class BaseFeatures(ABC):
    @property
    @abstractmethod
    def table_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def index_column(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def sql_code(self):
        raise NotImplementedError

    # Default property
    sql_asis = False
    run_checks = True
    database = os.environ.get("SNOWFLAKE_DATABASE", None)
    schema = os.environ.get("SNOWFLAKE_SCHEMA", None)
    resolution = None

    def __init__(self):
        if self.database is None:
            raise ValueError("DATABASE not specified. Please set the SNOWFLAKE_DATABASE env variable")

        if self.index_column is None:
            raise ValueError(f"Index column not specified for class {self.__class__.__name__}. Exiting!")

        if self.sql_code is None:
            raise ValueError(f"SQL Code not provided for feature set {self.__class__.__name__}! Exiting!")

    def run_sql(self, cnx, params: dict = {}) -> None:
        """
        Executes the SQL code and returns nothing. If it fails it throws an error.

        Inputs:
            cnx : snowflake.connection
            params : dict
                Optional parameters dictionary that changes values in the self.sql_code

        Returns:
            Nothing or an error
        """
        try:
            if self.sql_asis:
                sql_string = self.sql_code.format(**params)
            else:
                sql_string = f"CREATE TABLE IF NOT EXISTS \
                            {self.database.upper()}.{self.schema.upper()}.{self.table_name.upper()} AS ( \
                            {self.sql_code.format(**params)} ); "

            # Handling multi statement sql code:
            for statement in sql_string.split(";"):
                cnx.cursor().execute(statement)

            if self.run_checks:
                # Check that the indices are unique:
                n_entries = (
                    cnx.cursor()
                    .execute(
                        f"select count(*) from {self.database.upper()}.{self.schema.upper()}.{self.table_name.upper()}"
                    )
                    .fetchall()[0][0]
                )
                n_distinct_indices = (
                    cnx.cursor()
                    .execute(
                        f"select count(distinct {self.index_column}) from {self.database}.{self.schema}.{self.table_name}"
                    )
                    .fetchall()[0][0]
                )
                if n_entries != n_distinct_indices:
                    raise BadTable(n_entries, n_distinct_indices)

                # Check that the index is in the column specs:
                table_columns = self._get_columns(cnx)
                if self.index_column not in table_columns:
                    raise ValueError(f"Index column {self.index_column} not in the table column list: {table_columns}")

                # Get table's resolution (for subsequent checks)
                self._get_resolution(cnx)

        except BadTable:
            logger.exception(
                f"""Bad Table! The number of unique indices is not equal to the number of rows! \
            TableName: {self.database}.{self.schema}.{self.table_name}, Index: {self.index_column} \
            """
            )
            raise

        except ValueError:
            logger.error(
                f"Value error on: TableName: {self.database}.{self.schema}.{self.table_name}, Index: {self.index_column}"
            )
            raise

        except Exception as e:
            logger.exception(
                f"Feature Set:{self.__class__.__name__}: Unexpected error occured on table: {self.database}.{self.schema}.{self.table_name}",
                e,
            )
            raise

        else:
            self.executed_sql = sql_string
            logger.info(
                f"Feature Set:{self.__class__.__name__}:: Table {self.database}.{self.schema}.{self.table_name} created and/or verified!"
            )

        # Need an exception here that routes the error appropriately.

    def _get_resolution(self, cnx):
        """
        Gets the resolution of the cells
        """
        self.resolution = h3.h3_get_resolution(
            cnx.cursor()
            .execute(f"select {self.index_column} from {self.database}.{self.schema}.{self.table_name} limit 1;")
            .fetchall()[0][0]
        )

    def _table_exists(self, cnx):
        """
        Checks if Snowflake table exists
        """
        exec_string = f"select to_boolean(count(1)) from {self.database.upper()}.information_schema.tables where table_name = '{self.table_name.upper()}' and table_schema = '{self.schema.upper()}'"
        return list(cnx.cursor().execute(exec_string).fetchall()[0])[0]

    def _get_ddl(self, cnx):
        """
        Pulls ddl of the snowflake table
        """
        return (
            cnx.cursor()
            .execute(f"select get_ddl('table', '{self.database}.{self.schema}.{self.table_name}')")
            .fetchall()[0][0]
        )

    def _drop(self, cnx):
        """
        Drops a Snowflake table
        """
        try:
            cnx.execute(f"drop table {self.database}.{self.schema}.{self.table_name};")
        except Exception:
            logger.exception(f"Failed to drop table {self.database}.{self.schema}.{self.table_name}")
        else:
            logger.info(f"Table {self.database}.{self.schema}.{self.table_name} dropped!")

    def _get_columns(self, cnx):
        """
        Gets the columns in the database.
        """
        try:
            columns = (
                cnx.cursor()
                .execute(f"desc table {self.database.upper()}.{self.schema.upper()}.{self.table_name.upper()};")
                .fetchall()
            )

        except Exception:
            logger.exception(f"Failed to fetch columns for table: {self.database}.{self.schema}.{self.table_name}")
            raise

        else:
            return [p[0] for p in columns]

    def _get_index(self):
        """
        Returns the index of the table.
        """
        return self.index_column


class BaseLabels(BaseFeatures):
    sql_asis = True

    def __init__(self):
        super().__init__()
        self._check_sql()

    def _check_sql(self) -> None:
        # I asked chatgpt for this.... Find the last "select ... from" statement
        select_matches = re.finditer(r"\bSELECT\b.*?\bFROM\b", self.sql_code, re.DOTALL | re.IGNORECASE)
        last_select_statement = None
        for match in select_matches:
            last_select_statement = match.group(0)
        selected_columns = re.search("select(.*)from", last_select_statement, re.DOTALL | re.IGNORECASE).group(1)
        if (
            "H3_BLOCKS" not in selected_columns
            or "LABEL" not in selected_columns
            or "WEIGHT" not in selected_columns
        ):
            raise ValueError(f"Label {self.__class__.__name__} should output H3_BLOCKS, LABEL, and WEIGHT")
