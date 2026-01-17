
class EmbeddingGenerator:
    def __init__(self, embedding_client):
        self.client = embedding_client

    def embed(self, text: str) -> list:
        return self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

