from typing import Iterable, Optional, Union

from fastembed.text.text_embedding_base import TextEmbeddingBase
from fastembed.common.utils import iter_batch

from qdrant_client import QdrantClient, models
from qdrant_client.qdrant_fastembed import SUPPORTED_EMBEDDING_MODELS

import numpy as np
import httpx
import concurrent.futures

def batch_to_request(batch, model_name):
    return {
        'input' : batch, 
        'model' : model_name, 
        'encoding_format' : 'float'
        }

class APITextEmbedding(TextEmbeddingBase):
    def __init__(
        self,
        model_name: str,
        model_url: str,
        request_kwargs: Optional[dict], # additional kwargs to httpx.post, ie headers 
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        
        self.model_url = model_url
        self.request_kwargs = request_kwargs if request_kwargs is not None else {}
        
    def api_embed(self, documents, batch_size, parallel):
        
        if (parallel is None) or (parallel==0):
            parallel = 1

        # send `parallel` concurrent post requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            for batch in iter_batch(documents, batch_size):
                batch = batch_to_request(batch, self.model_name)
                future = executor.submit(httpx.post, self.model_url, json=batch, **self.request_kwargs)
                futures.append(future)

            concurrent.futures.wait(futures)

            for future in futures:
                response = future.result()
                response = response.json()
                response = [i['embedding'] for i in response['data']]
                response = np.array(response, dtype=np.float32)
                yield from response
        
    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, send `parallel` concurrent requests
                If None or <= 1, send one request at a time

        Returns:
            List of embeddings, one per document
        """
        
        yield from self.api_embed(documents, batch_size, parallel)

def add_api_model(
        model_name: str, 
        model_url: str, 
        request_kwargs: dict, 
        embedding_size: int, 
        distance_metric: models.Distance, 
        client: QdrantClient
    ):
    
    model = APITextEmbedding(model_name, model_url, request_kwargs)
    
    client._embedding_model_name = model_name
    client.embedding_models[model_name] = model
    
    SUPPORTED_EMBEDDING_MODELS[model_name] = (embedding_size, distance_metric)
