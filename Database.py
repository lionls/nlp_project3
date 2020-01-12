from flair.embeddings import ELMoEmbeddings
import os
import numpy as np
import torch
import sys
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

class Database(object):
    
    def __init__(self, docs):
        #self.documents_orig = np.loadtxt(docs, delimiter='\n', dtype = str)   # only 9999 documents
        self.documents_orig = []
        with open(docs, 'r') as f:     # getting 10k docs using this
            self.documents_orig = f.readlines()
        
        self.documents = []
        self.elmo = ELMoEmbeddings()
        #self.embedding = DocumentPoolEmbeddings([self.elmo])
        self.debug = True
    
    def knn(self, query, query_txt, k):
        #cos_sim = torch.mm(self.documents, query) / (torch.norm(query) * torch.norm(self.documents))

        cos_sim = torch.nn.functional.cosine_similarity(self.documents,query)
        
        topk, topk_indices = torch.topk(cos_sim, k, 0, True)
        
        topk_indices = topk_indices.numpy().astype('int')
        topk = topk.numpy().astype('float')
        top_combined = np.vstack((topk,topk_indices)).T
        
        if self.debug:
            print("\n")
            print("Query: ", query_txt, " | index: ", topk_indices.T)
            [print(self.documents_orig[int(i[1])], " --- ", i[0]) for i in top_combined]
        
        return list(zip(topk,topk_indices)) #used to return tuples
    
    def load_documents_into_embedding(self):
        print("Embedding ",len(self.documents_orig), " Documents")
        #self.documents_orig = self.documents_orig[0:50]
        self.documents = [self.elmo.embed(Sentence(elem)) for elem in self.documents_orig]
        
        self.documents = torch.stack([torch.cat([token.embedding.unsqueeze(0) for token in elem[0]], dim=0)[0] for elem in self.documents])
        
        np.save("./documents_embedded.npy",self.documents)
    
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
        
        #self.embedding.embed(sentence)
        
        self.elmo.embed(sentence)
        
        sentence = [token.embedding.unsqueeze(0) for token in sentence][0]
        
        #print(sentence)
        
        # A returned list should look like this for k=5. Btw. the numbers are made up!
        
        #[
        #            (0.89316645860672, 1567),
        #            (0.6174346804618835, 125),
        #            (0.5975501537321234, 1181),
        #            (0.5779426293373108, 3979),
        #            (0.5110726475715637, 7155),
        #        ]


        return self.knn(sentence, query, k=k)
            
            
            
    def run_query_txt(self,text):
        self.queries = np.loadtxt(text, delimiter='\n',dtype = str)
        
        results = []

        for query in self.queries:
            out = self.run_query(query)
            results.append(out)


        #saving results

        file = open("results.txt",'w')

        for elem in results:
            out = ""
            for res in elem:
                out += str(res[0]) + ","+ str(res[1]) + ";"
            out = out[:-1]
            out += '\n'
            file.write(out)

        file.close()


def main():
    data = Database(docs = "documents.txt")
    if not os.path.exists("./documents_embedded.npy"):
        data.load_documents_into_embedding()
    else:
        data.documents = torch.tensor(np.load("./documents_embedded.npy"))

    data.run_query_txt("queries.txt")

    for line in sys.stdin:
        if 'EXIT' == line.rstrip():
            break
        data.run_query(line)

    pass

if __name__ == "__main__":
    main()
