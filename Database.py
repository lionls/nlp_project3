from flair.embeddings import ELMoEmbeddings
import numpy as np
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

class Database(object):
    
    def __init__(self, docs):
        #self.glove = np.load("glove_word_embeddings.npy")
        self.documents_orig = np.loadtxt(docs, delimiter='\n',dtype = str)
        self.documents = []
        self.elmo = ELMoEmbeddings()
        self.embedding = DocumentPoolEmbeddings([self.elmo])
        self.debug = True
    
    def knn(self, query, k):
        cos_sim = np.dot(self.documents, query.embedding.data.numpy()) / (np.linalg.norm(query.embedding.data.numpy()) * np.linalg.norm(self.documents))
        k_best_indices = np.argpartition(cos_sim, -k)[-k:]
        combined = [(cos_sim[int(i)],i) for i in k_best_indices]
        combined.sort(key=lambda a: a[0], reverse=True)
        
        
        if self.debug:
            print("Query: ", query, " index: ", k_best_indices)
            [print(self.documents_orig[int(i[1])], " --- ", i[0]) for i in combined]
        
        return combined
    
    def load_documents_into_embedding(self):
        print("Embedding ",len(self.documents_orig), " Documents")
        self.documents = [Sentence(elem) for elem in self.documents_orig]

        #self.documents = self.documents[0:250] ##delete later just for faster testing
        
        self.embedding.embed(self.documents)
    
        self.documents = [elem.embedding.data.numpy() for elem in self.documents]
    
        print(self.documents)
    
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
        
        self.embedding.embed(sentence)
        
        # A returned list should look like this for k=5. Btw. the numbers are made up!
        
        #[
        #            (0.89316645860672, 1567),
        #            (0.6174346804618835, 125),
        #            (0.5975501537321234, 1181),
        #            (0.5779426293373108, 3979),
        #            (0.5110726475715637, 7155),
        #        ]


        return self.knn(sentence, k=k)
            
            
            
    def run_query_txt(self,text):
        self.queries = np.loadtxt(text, delimiter='\n',dtype = str)
        
        results = []

        for query in self.queries:
            out = self.run_query(query)
            results.append(out)

        print(results)

        #saving results

        file = open("results.txt",'w')

        for elem in results:
            out = ""
            for res in elem:
                out += str(res[0]) + ","+ str(res[1]) + ";"
            
            out += '\n'
            file.write(out)

        file.close()


def main():
    data = Database(docs = "documents.txt")
    data.load_documents_into_embedding()

    data.run_query_txt("queries.txt")
    pass

if __name__ == "__main__":
    main()
