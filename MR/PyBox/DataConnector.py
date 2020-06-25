import os
import struct
import pyodbc
from itertools import chain, repeat
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from inspect import ismethod


def bytes2mswin_bstr(value: bytes) -> bytes:
    """"Convert a sequence of bytes into a (MS-Windows) BSTR (as bytes)"""
    encoded_bytes = bytes(chain.from_iterable(zip(value, repeat(0))))
    return struct.pack("<i", len(encoded_bytes)) + encoded_bytes


class DataConnector:
    """
    Connects to SQL database or flat file and returns required analytical table.
    """

    def __init__(self, authenticator=None):

        if 'ODBC Driver 17 for SQL Server' in pyodbc.drivers():
            self._driver = 'ODBC Driver 17 for SQL Server'
        else:
            self._driver = '{SQL Server}'

        self.Trusted_Connection = 'Yes'
        self.server = os.environ["server"]
        self.schema = os.environ["schema"]
        self.database = os.environ["database"]
        self.driver = os.environ["driver"]

        self.authenticator = authenticator

    def get_token(self):
        if hasattr(self.authenticator, "get_token_for_azure_sql_db") \
                and ismethod(getattr(self.authenticator, "get_token_for_azure_sql_db")):

            return self.authenticator.get_token_for_azure_sql_db()

        else:
            print("No authenticator class has been provided. Accessing an Azure SQL database will fail.")
            return

    def connect_to_db(self, driver, server, database, trusted_connection):
        """
        Connects to SQL database
        :param driver: str, sql driver name
        :param server: str, server address
        :param database: str, name of the database to connect to
        :param trusted_connection: str, "yes" or "no"
        :return: connection
        """

        print("Driver: " + driver)

        if server.endswith(".munichre.com"):

            # on-prem server ends with ".munichre.com", no authentication w/ token necessary

            # connectionString = "DRIVER=" + driver \
            #                    + ";SERVER=" + server \
            #                    + ";DATABASE=" + database \
            #                    + ";Trusted_Connection=" + trusted_connection

            connectionString = f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection={trusted_connection}"
            quoted = quote_plus(connectionString)
            engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}", fast_executemany=True, echo=True)

        elif server.endswith(".database.windows.net"):

            # Azure SQL DB Servers end with ".database.windows.net", token to access DBs needs to be generated

            # At some point, the curly braces were removed from the driver string, b/c for on-prem DB access, they are not needed.
            # Here, however, they are and need to be added again.
            if (" " in driver) and (driver[0] != "{"):
                driver = "{" + driver + "}"

            connectionString = f"DRIVER={driver};SERVER={server};DATABASE={database}"
            token = self.get_token()
            tokenstruct = bytes2mswin_bstr(bytes(token, "UTF-8"))

            # SQL_COPT_SS_ACCESS_TOKEN is 1256; it's specific to msodbcsql driver so pyodbc does not have it defined
            SQL_COPT_SS_ACCESS_TOKEN = 1256

            # open connection
            conn = pyodbc.connect(connectionString, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: tokenstruct})
            engine = create_engine(f"mssql+pyodbc://?Trusted_Connection={trusted_connection}",
                                   fast_executemany=True,
                                   echo=True,
                                   creator=lambda: conn)

        else:
            raise NotImplementedError("Server address unknown.")

        return engine
