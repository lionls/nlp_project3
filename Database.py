from flair.embeddings import ELMoEmbeddings
import numpy as np
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

class Database(object):
    
    def __init__(self, docs):
        #self.glove = np.load("glove_word_embeddings.npy")
        self.documents = np.loadtxt(docs, delimiter='\n',dtype = str)
        self.elmo = ELMoEmbeddings()
        self.embedding = DocumentPoolEmbeddings([self.elmo])
    
    def knn(self, query, k):
        cos_sim = 1 - np.dot(self.documents, query.embedding.data.numpy()) / (np.linalg.norm(query.embedding.data.numpy()) * np.linalg.norm(self.documents))
        k_best_indices = np.argpartition(cos_sim,k)[:k]
    
        print(k_best_indices)
    #return topk, [cos[int(i)] for i in topk]
    
    def load_documents_into_embedding(self):
        print("Embedding ",len(self.documents), " Documents")
        self.documents = [Sentence(elem) for elem in self.documents]

        self.documents = self.documents[0:10] ##delete later just for faster testing
        
        self.embedding.embed(self.documents)
    
        self.documents = [elem.embedding.data.numpy() for elem in self.documents]
    
    
    def run_query(self, query, k=None):
        """Run a query on the given documents based on word embeddings
        
        Arguments:
            query {str} -- Query string.
        
        Keyword Arguments:
            k {int} -- The top documents to return (default: 10)
        
        Returns:
            list[tuple[float, int]] -- Sorted list of tuples, which contain the score and the document id.
                Made up example to show the formatting with k=5:
                        [(0.89316645860672, 1567), 
                        (0.6174346804618835, 125), 
                        (0.5975501537321234, 1181), 
                        (0.5779426293373108, 3979), 
                        (0.5110726475715637, 7155)]
        """
        if k is None:
            k = 10
        
        
        sentence = Sentence(query)
        
        self.elmo.embed(sentence)
        print(sentence.embedding.data.numpy())
        #print(self.knn(sentence,k=1))
        

        # A returned list should look like this for k=5. Btw. the numbers are made up!
        return [
            (0.89316645860672, 1567),
            (0.6174346804618835, 125),
            (0.5975501537321234, 1181),
            (0.5779426293373108, 3979),
            (0.5110726475715637, 7155),
        ]


def main():
    data = Database(docs = "documents.txt")
    #data.load_documents_into_embedding()
    
    data.run_query("hallo")
    pass

if __name__ == "__main__":
    main()
