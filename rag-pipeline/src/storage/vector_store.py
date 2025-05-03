class VectorStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self.vectors = []

    def save_to_disk(self):
        # Code to save vectors to disk using ChromaDB
        pass

    def load_from_disk(self):
        # Code to load vectors from disk using ChromaDB
        pass

    def add_vector(self, vector):
        self.vectors.append(vector)

    def get_vectors(self):
        return self.vectors