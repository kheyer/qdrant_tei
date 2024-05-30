# Qdrant + TEI

This repo enables linking the [Qdrant Client](https://github.com/qdrant/qdrant-client/tree/master) with [Fastembed](https://github.com/qdrant/fastembed/tree/main) to any embedding API that follows the OpenAI embedding schema.

The repo also contains a docker-compose file for spinning up [Qdrant](https://github.com/qdrant) with the Huggingface [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) server.


## Use Qdrant with TEI

Start the docker-compose system

```
docker-compose up -d --build
```

On the client side, install dependencies: 

```
pip install -r requirements.txt
```

Patch the Qdrant client, then use normally

```python
from api_embedding import APITextEmbedding, add_api_model
from qdrant_client import QdrantClient, models

# create client
client = QdrantClient("localhost", port=6333)

# model parameters, for example the OpenAI Ada embedding model
url = 'http://localhost:8100/embeddings'
model_name = 'BAAI/bge-large-en-v1.5'
embedding_size = 768
distance_metric = models.Distance.COSINE
request_kwargs = None

# patch qdrant client
add_api_model(
    model_name, 
    url, 
    request_kwargs, 
    embedding_size, 
    distance_metric, 
    client
    )

# add documents
docs = ["sample doc 1", "sample doc 2"]
ids = [0, 1]

client.add(
    collection_name="demo_collection",
    documents=docs,
    ids=ids
)

search_result = client.query(
    collection_name="demo_collection",
    query_text="This is a query document"
)
print(search_result)
```

## Use TEI with Multiple GPUs

To use this system with multiple GPUs, update `docker-compose.yml` and `nginx.conf`

In `docker-compose.yml`, add replicas of the `tei-0` service and assign each replica 
a different GPU. Currently, TEI only uses a single GPU, so do not assign multiple GPUs to 
a given replica. The `docker-compose.yml` file currently contains a commented-out `tei-1` 
service as an example.

In `nginx.conf`, add the new TEI replicas to the upstream section. The `nginx.conf` file 
currently contains a commented-out `tei-1` service as an example


## Use Qdrant with TEI any OpenAI Compatible Embedding Endpoint

The client-side code in this repo works with any embedding endpoint 
that follows the OpenAI schema. To use an external endpoint with the 
Qdrant client, we just need to pass additional request arguments such 
as a header with an auth key.

On the cient side, install dependencies:

```
pip install -r requirements.txt
```

Patch the Qdrant client, then use normally

```python
from api_embedding import APITextEmbedding, add_api_model
from qdrant_client import QdrantClient, models

# create client
client = QdrantClient("localhost", port=6333)

# model parameters, for example the OpenAI Ada embedding model
url = "https://api.openai.com/v1/embeddings"
model_name = 'text-embedding-ada-002'
embedding_size = 1536
distance_metric = models.Distance.COSINE

# create additional kwargs for request, in this case OpenAI key/header
openai_api_key = ...
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
request_kwargs = {'headers' : headers}

# patch qdrant client
add_api_model(
    model_name, 
    url, 
    request_kwargs, 
    embedding_size, 
    distance_metric, 
    client
    )

# add documents
docs = ["sample doc 1", "sample doc 2"]
ids = [0, 1]

client.add(
    collection_name="demo_collection",
    documents=docs,
    ids=ids
)

search_result = client.query(
    collection_name="demo_collection",
    query_text="This is a query document"
)
print(search_result)
```




