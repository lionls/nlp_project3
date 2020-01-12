import numpy as np
from sklearn import neighbors


class PuzzleSolver(object):
    
    def __init__(self):
        self.glove = np.load("glove_word_embeddings.npy")
        #self.words = np.loadtxt("words.txt", delimiter='\n',dtype = str)
        #self.words  = np.append(self.words, [" "] * 31) # fix because length is too short
    
        with open("words.txt", 'r') as f:     # getting all the words like this
            self.words = np.array([line.rstrip() for line in f.readlines()])

 
    def get_glove_vector(self, word):
        return self.glove[np.where(self.words == word)]
    
    def solve_puzzle(self, a,b,c):
        """Solve an analogy puzzle.
        
        Arguments:
            a {str} -- is to
            b {str} -- like
            c {str} -- is to
        
        Returns:
            str -- Token that solves the puzzle
        """
        # to implement
        aVec = self.get_glove_vector(a)
        bVec = self.get_glove_vector(b)
        cVec = self.get_glove_vector(c)
        
        xVec = bVec - aVec + cVec
        
        ## calculate cosine similarity with k = 1
        
        k = 5

        cos_sim = 1 - np.dot(self.glove, xVec[0]) / (np.linalg.norm(xVec) * np.linalg.norm(self.glove))

        k_best_indices = np.argpartition(cos_sim,k)[:k]
  
        out = self.words[k_best_indices]
        print("5 most similar: ", out)
        ## make sure its not the same
        j=k
        while(k>0):
            if(out[j-k] != b):
                x = out[j-k]
                break;
            else:
                k -= 1
        print("result: ", x)
        return x

def test_solve_puzzle(puzzle_solver):
    # This test shall pass!
    assert puzzle_solver.solve_puzzle('man', 'king', 'woman') == 'queen'
    assert puzzle_solver.solve_puzzle('bank', 'money', 'stock') == 'profits' #1
    assert puzzle_solver.solve_puzzle('flying', 'running', 'fly') == 'run' #2
    assert puzzle_solver.solve_puzzle('driving', 'flying', 'car') == 'aircraft' #3
    assert puzzle_solver.solve_puzzle('son', 'man', 'daughter') == 'woman' #4
    assert puzzle_solver.solve_puzzle('man', 'he', 'woman') == 'she' #5
    assert puzzle_solver.solve_puzzle('water', 'ship', 'air') == 'aircraft' #6
    assert puzzle_solver.solve_puzzle('computer', 'programmer', 'music') == 'composer' #7

if __name__ == "__main__":
    puzzle_solver = PuzzleSolver()
    test_solve_puzzle(puzzle_solver)
