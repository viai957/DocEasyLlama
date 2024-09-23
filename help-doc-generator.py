import os
import numpy as np
import mimetypes
from typing import List, Dict, Any, Generator, Tuple, Optional
import hashlib
from pathlib import Path
import logging
from tenacity  import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    SparseVector,
    PointStruct,
    SearchRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
)

import csv
import json
import xml.etree.ElementTree as ET
import json
import markdown

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()  # Console handler
file_handler = logging.FileHandler('processing.log', mode='w')  # File handler

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# logging handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Defining Embedding model
dense_model_name = "BAAI/bge-large-en-v1.5"
sparse_model_name = "prithvida/Splade_PP_en_v1"

class FileHandler:
    @staticmethod
    def read_file(file_path: str) -> str:
        """Read content from various file types."""
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type is None:
            if file_path.endswith(".md"):
                mime_type = "text/markdown"
            elif file_path.endswith(".in"):
                mime_type = "text/plain"  # Example for .in files
            else:
                raise ValueError(f"Unknown file type: {file_path}")
        
        if mime_type.startswith("text"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif mime_type == "text/markdown":
            return FileHandler._read_markdown(file_path)
        elif mime_type.endswith("xml"):
            return FileHandler._read_xml(file_path)
        elif mime_type.endswith("py"):
            return FileHandler._read_python(file_path)
        elif mime_type.endswith("json"):
            return FileHandler._read_json(file_path)
        elif mime_type.endswith("md"):
            return FileHandler._read_markdown(file_path)
        elif mime_type.endswith("csv"):
            return FileHandler._read_csv(file_path)
        elif mime_type.endswith("in"):
            return FileHandler._read_in(file_path)
        else:
            raise ValueError(f"Unknown file type: {mime_type}")
    
    @staticmethod
    def _read_python(file_path: str) -> str:
        """Read content from a Python (.py) file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _read_markdown(file_path: str) -> str:
        """Read content from a Markdown (.md) file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _read_in(file_path: str) -> str:
        """Read content from a `.in` file (typically plain text or config)."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _read_json(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return json.dumps(data, indent=2)

    @staticmethod
    def _read_csv(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            return '\n'.join(','.join(row) for row in reader)

    @staticmethod
    def _read_xml(file_path: str) -> str:
        tree = ET.parse(file_path)
        return ET.tostring(tree.getroot(), encoding='unicode', method='xml')
    
    @staticmethod
    def _read_markdown(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file .read()
        return markdown.markdown(content)
    
class TextChunker:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 100000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

class EmbeddingGenerator:
    def __init__(self):
        self.dense_model = TextEmbedding(model_name=dense_model_name, batch_size=32)
        self.sparse_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1", batch_size=32)

    def generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for the list of texts."""
        return list(self.dense_model.embed(texts))

    def generate_sparse_embeddings(self, texts: List[str]) -> List[SparseEmbedding]:
        """Generate sparse embeddings for the list of texts."""
        return list(self.sparse_model.embed(texts))

    
class VectorDBHandler:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)

    def create_collection(self, dense_vector_size: int, sparse_model_name: str, distance: models.Distance = models.Distance.COSINE):
        """
        Create a new collection in Qdrant.
        
        Args:
            vector_size (int): The size of the vector embeddings.
            distance (models.Distance): The distance metric to use (default: COSINE).
        
        Raises:
            UnexpectedResponse: If there's an error creating the collection.
        """
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": VectorParams(
                        size=dense_vector_size,
                        distance=distance
                    )
                },
                sparse_vectors_config={
                    "text-sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,  # Move `on_disk` here into SparseIndexParams
                        )
                    )
                },
            )
            self.logger.info(f"Collection '{self.collection_name}' created successfully with dense and sparse vectors.")
        
        except UnexpectedResponse as e:
            self.logger.error(f"Failed to create collection '{self.collection_name}': {str(e)}")
            raise

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except UnexpectedResponse:
            return False

    def delete_collection(self):
        """Delete the collection if it exists."""
        if self.collection_exists():
            try:
                self.client.delete_collection(self.collection_name)
                self.logger.info(f"Collection '{self.collection_name}' deleted successfully.")
            except UnexpectedResponse as e:
                self.logger.error(f"Failed to delete collection '{self.collection_name}': {str(e)}")
                raise
        else:
            self.logger.warning(f"Collection '{self.collection_name}' does not exist. Nothing to delete.")

    def add_documents(self, documents: List[str], sparse_embeddings: List[SparseEmbedding], dense_embeddings: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]):
        """
        Add documents with embeddings and metadata to the collection.
        
        Args:
            documents (List[str]): List of document texts.
            embeddings (List[List[float]]): List of embedding vectors.
            metadata (List[Dict[str, Any]]): List of metadata dictionaries.
            ids (List[str]): List of unique identifiers for the documents.
        
        Raises:
            ValueError: If the input lists have mismatched lengths.
            UnexpectedResponse: If there's an error adding the documents.
        """
        if not (len(documents) == len(sparse_embeddings) == len(dense_embeddings) == len(metadata) == len(ids)):
            raise ValueError("All input lists must have the same length.")
        
        try:
            points = []
            for idx in range(len(documents)):
                sparse_vectors = SparseVector(
                    indices=sparse_embeddings[idx].indices.tolist(),
                    values=sparse_embeddings[idx].values.tolist(),
                )
                dense_vector = dense_embeddings[idx]
                point = PointStruct(
                    id=ids[idx],
                    payload=metadata[idx],
                    vector = {
                        "text-dense": dense_vector,
                        "text-sparse": sparse_vectors
                    },
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
                )
            self.logger.info(f"Successfully added {len(documents)} documents to collection '{self.collection_name}'.")
        except UnexpectedResponse as e:
            self.logger.error(f"Failed to add documents to collection '{self.collection_name}': {str(e)}")
            raise

    def hybrid_search(self, query_sparse_vector: SparseEmbedding, query_dense_vector: List[float], limit: int = 10, query_filter: Optional[Dict] = None) -> Tuple[List[ScoredPoint], List[ScoredPoint]]:
        """
        Search for similar documents in the collection using a hybrid query.
        
        Args:
            query_vector (SparseEmbedding): The sparse query vector to search for.
            query_dense_vector (List[float]): The dense query vector to search for.
            limit (int): The maximum number of results to return (default: 10).
            query_filter (Optional[Dict]): Optional filter to apply to the search (default: None).
        
        Returns:
            List[Dict[str, Any]]: List of search results.
        
        Raises:
            UnexpectedResponse: If there's an error during the search operation.
        """
        try:
            search_results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=[
                    SearchRequest(
                        vector=NamedVector(
                            name="text-dense",
                            vector=query_dense_vector,
                        ),
                        limit=limit,
                        with_payload=True,
                    ),
                    SearchRequest(
                        vector=NamedSparseVector(
                            name="text-sparse",
                            vector=SparseVector(
                                indices=query_sparse_vector.indices.tolist(),
                                values=query_sparse_vector.values.tolist(),
                            ),
                        ),
                        limit=limit,
                        with_payload=True,
                    ),
                ],
            )
            
            # Return search results for dense and sparse separately
            return search_results[0], search_results[1]
        
        except UnexpectedResponse as e:
            self.logger.error(f"Search operation failed in collection '{self.collection_name}': {str(e)}")
            raise
    
    
    def rrf(self, rank_lists: List[List[Tuple[str, int]]], alpha=60, default_rank=1000) -> List[Tuple[str, float]]:
        """ 
        Implementation of Reciprocal Rank Fusion (RRF) algorithm to combine results from sparse
        and dense vector searches.

        Args:
            rank_lists: List of ranked lists (each list is a list of (id, rank) tuples).
            alpha: Parameter to control the importance of the dense vector search results.
            default_rank: Default rank value for documents not present in the ranked lists.

        Returns:
            List[Tuple[str, float]]: Combined ranked results with RRF scores.
        """
        all_items = set(item for rank_list in rank_lists for item, _ in rank_list)
        item_to_index = {item: idx for idx, item in enumerate(all_items)}
        rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)
        
        for list_idx, rank_list in enumerate(rank_lists):
            for item, rank in rank_list:
                rank_matrix[item_to_index[item], list_idx] = rank
        
        rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)
        sorted_indices = np.argsort(-rrf_scores)  # Sort in descending order
        
        sorted_items = [(list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices]
        return sorted_items
        
    def rank_results(self, sparse_results: List[ScoredPoint], dense_results: List[ScoredPoint]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank results using RRF after performing hybrid search.
        sparse_results: List of results from sparse search.
        dense_results: List of results from dense search.
        """
        # Convert the sparse and dense search results into ranked lists (id, rank)
        sparse_rank_list = self._rank_list(sparse_results)
        dense_rank_list = self._rank_list(dense_results)

        # Rank them using RRF (Reciprocal Rank Fusion)
        ranked_results = self.rrf([sparse_rank_list, dense_rank_list])
        
        # Return ranked results
        return ranked_results


    def _rank_list(self, search_result: List[ScoredPoint]):
        """
        Convert search results into a list of (id, rank) tuples.
        """
        return [(point.id, rank + 1) for rank, point in enumerate(search_result)]

    def search(self, query_vector: List[float], limit: int = 5, query_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the collection.
        
        Args:
            query_vector (List[float]): The query vector to search for.
            limit (int): The maximum number of results to return (default: 5).
            query_filter (Optional[Dict]): Optional filter to apply to the search (default: None).
        
        Returns:
            List[Dict[str, Any]]: List of search results.
        
        Raises:
            UnexpectedResponse: If there's an error during the search operation.
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter
            )
            return [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in results]
        except UnexpectedResponse as e:
            self.logger.error(f"Search operation failed in collection '{self.collection_name}': {str(e)}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dict[str, Any]: Collection information.
        
        Raises:
            UnexpectedResponse: If there's an error retrieving collection info.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vector_size": info.config.params.vector_size,
                "distance": info.config.params.distance,
                "points_count": info.points_count
            }
        except UnexpectedResponse as e:
            self.logger.error(f"Failed to get info for collection '{self.collection_name}': {str(e)}")
            raise

    def delete_points(self, point_ids: List[str]):
        """
        Delete specific points from the collection.
        
        Args:
            point_ids (List[str]): List of point IDs to delete.
        
        Raises:
            UnexpectedResponse: If there's an error deleting points.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            self.logger.info(f"Successfully deleted {len(point_ids)} points from collection '{self.collection_name}'.")
        except UnexpectedResponse as e:
            self.logger.error(f"Failed to delete points from collection '{self.collection_name}': {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_payload_index(self, field_name: str, field_schema: models.PayloadSchemaType):
        """Create an index on a payload field."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
            self.logger.info(f"Successfully created index on field '{field_name}' in collection '{self.collection_name}'.")
        except UnexpectedResponse as e:
            self.logger.error(f"Failed to create index on field '{field_name}' in collection '{self.collection_name}': {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def update_payload(self, point_id: str, payload: Dict[str, Any]):
        """Update the payload of a specific point."""
        try:
            self.client.update_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[point_id]
            )
            self.logger.info(f"Successfully updated payload for point '{point_id}' in collection '{self.collection_name}'.")
        except UnexpectedResponse as e:
            self.logger.error(f"Failed to update payload for point '{point_id}' in collection '{self.collection_name}': {str(e)}")
            raise

class DocumentProcessor:
    def __init__(self, vector_db_handler: VectorDBHandler, embedding_generator: EmbeddingGenerator):
        self.vector_db_handler = vector_db_handler
        self.embedding_generator = embedding_generator
        self.file_handler = FileHandler()
        self.chunker = TextChunker()
    
    def print_directory_structure(self, directory_path: str, indent: str = '') -> None:
        """Recursively print the directory structure."""
        for root, dirs, files in os.walk(directory_path):
            # Calculate the level of the current directory to indent
            level = root.replace(directory_path, '').count(os.sep)
            indent = '│   ' * level + '├── ' if level > 0 else ''
            print(f"{indent}{os.path.basename(root)}/")
            
            subindent = '│   ' * (level + 1) + '├── '
            for file in files:
                print(f"{subindent}{file}")
            # Prevent walking into subdirectories multiple times
            dirs[:] = []

    def process_directory(self, directory_path: str) -> None:
        """Process all files in a directory and its subdirectories and print the directory structure."""
        # First, print the directory structure
        self.print_directory_structure(directory_path)

        # Process all files in the directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    self.process_file(file_path)
                    # Print when the file is processed successfully ans stored in Vector DB
                    print(f"{file_path} -> embedded -> Vector DB")
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {str(e)}")

    def process_file(self, file_path: str) -> None:
        """Process a single file."""
        try:
            # Read the file content
            content = self.file_handler.read_file(file_path)

            # Chunk the content for large files
            chunks = self.chunker.chunk_text(content)

            # Generate embeddings for each chunk
            dense_embeddings = self.embedding_generator.generate_dense_embeddings(chunks)
            
            # Generate metadata and IDs
            metadata = self._generate_metadata(file_path)
            ids = self._generate_ids(chunks, file_path)

            # Include chunk content in metadata
            for i, chunk in enumerate(chunks):
                metadata[i]["chunk_content"] = chunk  # Add the actual chunk content to the metadata
            
            # Create dummy sparse embeddings if needed (assuming no sparse embeddings for now)
            sparse_embeddings = [SparseEmbedding(np.zeros(len(chunks)), np.zeros(len(chunks))) for _ in chunks]

            # Add documents to the vector database with the embeddings, metadata, and ids
            self.vector_db_handler.add_documents(
                documents=chunks,
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                metadata=metadata,  # Pass metadata with chunk content included
                ids=ids  # Pass the generated IDs here
            )
            logging.info(f"Successfully processed file: {file_path}")
            print(f"{file_path} -> embedded -> Vector DB")

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")


    def _generate_metadata(self, file_path: str) -> List[Dict[str, Any]]:
        """Generate metadata for chunks from a file."""
        file_stats = os.stat(file_path)
        base_metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": file_stats.st_size,
            "last_modified": file_stats.st_mtime,
            "mime_type": mimetypes.guess_type(file_path)[0]
        }
        return [base_metadata.copy() for _ in range(len(self.chunker.chunk_text(self.file_handler.read_file(file_path))))]

    def _generate_ids(self, chunks: List[str], file_path: str) -> List[str]:
        """Generate unique IDs for chunks."""
        return [hashlib.md5(f"{file_path}_{i}_{chunk[:50]}".encode()).hexdigest() 
                for i, chunk in enumerate(chunks)]

class QueryProcessor:
    def __init__(self, vector_db_handler: VectorDBHandler, embedding_generator: EmbeddingGenerator):
        self.vector_db_handler = vector_db_handler
        self.embedding_generator = embedding_generator

    def process_query(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Process a query and return relevant chunks with metadata."""
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        results = self.vector_db_handler.search(query_embedding, limit)
        return results
    
    def process_hybrid_query(self, query: str, limit: int = 10):
        """
        Process a hybrid query, perform both sparse and dense searches, and rank results using RRF.
        """
        sparse_embedding = self.embedding_generator.generate_sparse_embeddings([query])[0]
        dense_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
        search_results = self.vector_db_handler.hybrid_search(sparse_embedding, dense_embedding, limit)
        
        ranked_results = self.vector_db_handler.rank_results(search_results[1], search_results[0])
        
        return ranked_results

class VectorDBSystem:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.vector_db_handler = VectorDBHandler(qdrant_client, collection_name)
        self.embedding_generator = EmbeddingGenerator()
        self.document_processor = DocumentProcessor(self.vector_db_handler, self.embedding_generator)
        self.query_processor = QueryProcessor(self.vector_db_handler, self.embedding_generator)

    def initialize_collection(self):
        """Initialize the vector database collection with both dense and sparse vectors."""
        self.vector_db_handler.create_collection(dense_vector_size=1024, sparse_model_name="text-sparse")

    def index_directory(self, directory_path: str):
        """Index all documents in a directory."""
        self.document_processor.process_directory(directory_path)

    def hybrid_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search that combines both dense and sparse embeddings.
        Results are ranked using Reciprocal Rank Fusion (RRF).
        """
        # Generate both sparse and dense embeddings for the query
        query_sparse_vector = self.embedding_generator.generate_sparse_embeddings([query])[0]
        query_dense_vector = self.embedding_generator.generate_dense_embeddings([query])[0]

        # Use the handler's hybrid_search to perform the search
        sparse_results, dense_results = self.vector_db_handler.hybrid_search(query_sparse_vector, query_dense_vector, limit)

        # Rank the results using Reciprocal Rank Fusion
        ranked_results = self.vector_db_handler.rank_results(sparse_results, dense_results)
        
        return ranked_results
    
# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Qdrant client (adjust parameters as needed)
    client = QdrantClient(host="localhost", port=6333)
    
    # Initialize the VectorDBSystem
    vector_db_system = VectorDBSystem(client, "test-v11-hybrid")
    
    # Initialize the collection
    vector_db_system.initialize_collection()
    
    # Index a directory
    vector_db_system.index_directory("./razorpay-python")
    
    # Perform a hybrid search
    query = "API for Settlements"
    results = vector_db_system.hybrid_search(query, limit=10)

    for result in results:
        # Assuming result is a tuple (id, score), fetch the corresponding point
        point_id, score = result
        point = vector_db_system.vector_db_handler.client.retrieve(vector_db_system.vector_db_handler.collection_name, ids=[point_id])
        
        # Printing results
        if point:
            print(f"Score: {score}")
            print(f"Metadata: {point[0].payload}")
            print(f"Content: {point[0].payload.get('chunk_content', 'N/A')}")  # This should now display the actual content
            print("---")


