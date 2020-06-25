import logging
import time

import pandas as pd
from sqlalchemy import text

from PyBox.DataConnector import DataConnector


class DataPublisher:
    """
    This class utilizes a DataConnector instance and interacts with the underlying data source.
    """

    def __init__(self, data_connector=None, authenticator=None):

        if data_connector is None:
            data_connector = DataConnector(authenticator=authenticator)

        self.data_connector = data_connector

    def get_table(self, table_name, number_rows="*", q='SELECT ' + "*" + ' FROM [{}].[{}].[{}];', use_h2o=None):
        """
        Get the full data table from the data source.
        :param table_name: String, Has to the be exact name of a table existing in the database.
        :return: A pandas DataFrame

        """

        driver = self.data_connector._driver
        server = self.data_connector.server
        database = self.data_connector.database
        trusted = self.data_connector.Trusted_Connection

        engine = self.data_connector.connect_to_db(driver=driver, server=server,
                                                   database=database, trusted_connection=trusted)

        connection = engine.connect()

        if number_rows == "*":
            query = q.format(database, self.data_connector.schema, table_name)
        else:
            query = 'SELECT ' + number_rows + ' FROM [{}].[{}].[{}];'.format(database, self.data_connector.schema, table_name)

        start = time.time()
        table = pd.read_sql_query(sql=query, con=connection)
        end = time.time()
        fit_time = end - start
        print("loading took " + str(fit_time) + " seconds")
        logging.info('Successfully fetched table {} from schema {}'.format(table_name, database))
        return table

    def publish_table(self, df, table_name, if_exists='fail'):
        """
        Publish a data frame as table in the data source.
        :param df: A pandas DataFrame
        :param table_name: String, the name of the table to be created in the data source.
        :param if_exists: Boolean, How to handle the case if the table already exists?
        Options are: replace, fail, append
        :param chunksize: int, Rows will be written in batches of this size at a time.
        :return: None

        """
        driver = self.data_connector._driver
        server = self.data_connector.server
        database = self.data_connector.database
        trusted = self.data_connector.Trusted_Connection

        engine = self.data_connector.connect_to_db(driver=driver, server=server,
                                                   database=database, trusted_connection=trusted)

        assert isinstance(df, pd.DataFrame)

        start = time.time()

        df.to_sql(name=table_name,
                  con=engine,
                  schema=self.data_connector.schema,
                  if_exists=if_exists,
                  index=False)

        end = time.time()
        fit_time = end - start
        print("writing took " + str(fit_time) + " seconds")
        logging.info('Successfully created table {} in schema {}'.format(table_name, self.data_connector.database))

    def shoot_sql(self, query_file):

        driver = self.data_connector._driver
        server = self.data_connector.server
        database = self.data_connector.database
        trusted = self.data_connector.Trusted_Connection

        engine = self.data_connector.connect_to_db(driver=driver, server=server,
                                                   database=database, trusted_connection=trusted)

        connection = engine.connect()

        with open(query_file, "r") as file:
            query = file.read()
            connection.execute(text(query))
