"""Loading logic for loading documents from an Azure CosmosDB database container."""
import json
from typing import Any, List, Optional, Callable, Dict

from langchain.docstore.document import Document

from langchain.document_loaders.base import BaseLoader


class AzureCosmosDBLoader(BaseLoader):
    """Loading Documents from CosmosDB."""

    def __init__(
            self, 
            conn_str: str, 
            database: str, 
            container: str, 
            query: str, 
            jq_schema: str,            
            enable_cross_partition_query: bool = True,
            text_content: bool = True
        ):
        """Initialize the AzureCosmosDBLoader.

        Args:
            conn_str (str): Connection string for Azure CosmosDB.
            database (str): Database id (name).
            container (str): Container id (name).
            query (str): CosmosDB items query.
            enable_cross_partition_query (bool): Enable cross-partition query.
            jq_schema (str): The jq schema to use to extract the data or text from
                the JSON.
            text_content (bool): Boolean flag to indicate whether the content is in
                string format, default to True.
        """
        try:
            import jq  # noqa:F401
        except ImportError:
            raise ImportError(
                "jq package not found, please install it with `pip install jq`"
            )
        
        self._conn_str = conn_str
        self.database = database
        self.container = container
        self.query = query
        self._enable_cross_partition_query = enable_cross_partition_query
        self._jq_schema = jq.compile(jq_schema)
        self._text_content = text_content

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from azure.cosmos import CosmosClient
        except ImportError as exc:
            raise ImportError(
                "Could not import azure cosmos python package. "
                "Please install it with `pip install azure-cosmos`."
            ) from exc

        docs: List[Document] = []
        db = CosmosClient.from_connection_string(conn_str=self._conn_str)
        database = db.get_database_client(database=self.database)
        container = database.get_container_client(container=self.container)

        for item in container.query_items(
            query=self.query,
            enable_cross_partition_query=self._enable_cross_partition_query
        ):
            self._parse(item, docs)
        return docs
    
    def _parse(self, content: str, docs: List[Document]) -> None:
        """Convert given content to documents."""
        data = self._jq_schema.input(json.loads(content))

        for i, sample in enumerate(data, len(docs) + 1):
            metadata = dict(
                source=str(self.file_path),
                seq_num=i,
            )
            text = self._get_text(sample=sample, metadata=metadata)
            docs.append(Document(page_content=text, metadata=metadata))

    def _get_text(self, sample: Any, metadata: dict) -> str:
        """Convert sample to string format"""
        content = sample

        if self._text_content and not isinstance(content, str):
            raise ValueError(
                f"Expected page_content is string, got {type(content)} instead. \
                    Set `text_content=False` if the desired input for \
                    `page_content` is not a string"
            )

        # In case the text is None, set it to an empty string
        elif isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content) if content else ""
        else:
            return str(content) if content is not None else ""
