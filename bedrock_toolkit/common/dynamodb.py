""" Common DynamoDB utility functions """
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

import boto3
from botocore.exceptions import ClientError, WaiterError

from bedrock_toolkit.logger_manager import LoggerManager

logger = LoggerManager.get_logger()

TableState = Literal['ACTIVE', 'CREATING', 'UPDATING', 'DELETING', 'NOT_EXISTS']

class TableLock:
    def __init__(self, table, lock_key, ttl_seconds=60) -> None:
        self.table = table
        self.lock_key = lock_key
        self.ttl_seconds = ttl_seconds

    def __enter__(self) -> None:
        expiration_time = int((datetime.now() + timedelta(seconds=self.ttl_seconds)).timestamp())
        retry_count = 0
        while retry_count < 5:
            try:
                self.table.put_item(
                    Item={
                        'chat_id': self.lock_key,
                        'locked': True,
                        'expiration': expiration_time
                    },
                    ConditionExpression='attribute_not_exists(chat_id) OR expiration < :now',
                    ExpressionAttributeValues={':now': int(datetime.now().timestamp())}
                )
                return
            except ClientError as e:
                if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
                    raise
                time.sleep(1)
                retry_count += 1
        raise Exception("Failed to acquire lock after 5 attempts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.table.delete_item(Key={'chat_id': self.lock_key})

class DynamoDBBase:
    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        delete_existing_table: bool = False,
        chat_id: str = "00000",
    ) -> None:
        self.table_name = table_name
        self.region = region
        self.delete_existing_table = delete_existing_table
        self.chat_id = chat_id

        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.table = self.dynamodb.Table(self.table_name)
        self.lock_key = 'table_lock'

        # Ensure the DynamoDB table exists, delete it first if the option is set
        self.ensure_table_exists(delete_existing_table)

    def _table_lock(self):
        return TableLock(self.table, self.lock_key)

    def ensure_table_exists(self, delete_existing: bool = False) -> None:
        max_retries = 30
        retry_delay = 3

        for attempt in range(max_retries):
            try:
                state = self._get_table_state()
                logger.info(f"Current table state: {state}")

                if delete_existing:
                    logger.info("Delete existing table flag is set to True")
                    if state in ['ACTIVE', 'CREATING', 'UPDATING']:
                        logger.info(f"Attempting to delete existing table {self.table_name}")
                        self._delete_table()
                        self._wait_for_table_deletion()
                    elif state == 'DELETING':
                        logger.info(f"Table {self.table_name} is already in DELETING state")
                        self._wait_for_table_deletion()
                    self._create_table()
                    self._wait_for_table_creation()
                    return
                elif state == 'ACTIVE':
                    logger.info(f"Table {self.table_name} is ready for use")
                    return
                elif state in ['CREATING', 'UPDATING']:
                    logger.info(f"Table {self.table_name} is in {state} state, waiting for it to become active")
                    self._wait_for_table_creation()
                    return
                elif state == 'NOT_EXISTS':
                    logger.info(f"Table {self.table_name} does not exist, creating it")
                    self._create_table()
                    self._wait_for_table_creation()
                    return

            except ClientError as e:
                logger.error(f"Error in ensure_table_exists: {e}")
                if attempt == max_retries - 1:
                    raise
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        raise Exception(f"Failed to ensure table exists after {max_retries} attempts")

    def _get_table_state(self) -> TableState:
        try:
            response = self.dynamodb.meta.client.describe_table(TableName=self.table_name)
            return response['Table']['TableStatus']
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return 'NOT_EXISTS'
            raise

    def _create_table(self) -> None:
        logger.info(f"Creating table {self.table_name}")
        try:
            self.dynamodb.meta.client.create_table(
                TableName=self.table_name,
                KeySchema=self._get_key_schema(),
                AttributeDefinitions=self._get_attribute_definitions(),
                BillingMode='PAY_PER_REQUEST'
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceInUseException':
                raise

    def _delete_table(self) -> None:
        logger.info(f"Deleting table {self.table_name}")
        try:
            self.dynamodb.meta.client.delete_table(TableName=self.table_name)
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                raise

    def _wait_for_table_creation(self) -> None:
        logger.info(f"Waiting for table {self.table_name} to become active")
        waiter = self.dynamodb.meta.client.get_waiter('table_exists')
        try:
            waiter.wait(TableName=self.table_name, WaiterConfig={'Delay': 5, 'MaxAttempts': 20})
            logger.info(f"Table {self.table_name} is now active")
        except WaiterError as e:
            logger.error(f"Error waiting for table to become active: {e}")
            raise

    def _wait_for_table_deletion(self) -> None:
        logger.info(f"Waiting for table {self.table_name} to be deleted")
        waiter = self.dynamodb.meta.client.get_waiter('table_not_exists')
        try:
            waiter.wait(TableName=self.table_name, WaiterConfig={'Delay': 5, 'MaxAttempts': 20})
            logger.info(f"Table {self.table_name} has been deleted")
        except WaiterError as e:
            logger.error(f"Error waiting for table to be deleted: {e}")
            raise

    def _convert_floats_to_decimal(self, obj):
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimal(i) for i in obj]
        return obj

    def _convert_decimals_to_float(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimals_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimals_to_float(i) for i in obj]
        return obj

    def _get_key_schema(self):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _get_key_schema")

    def _get_attribute_definitions(self):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _get_attribute_definitions")
