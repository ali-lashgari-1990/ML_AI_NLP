from sklearn.decomposition import PCA
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText

class hw3:
    # ques1_3
    def get_word2vec_matrix(self,  words_list ):
        # first sort words_list and then complete the function

        return M, word2Ind, non_exist

    # ques1_6    
    def get_glove_matrix(self,  words_list ):
        # first sort words_list and then complete the function

        return M, word2Ind, non_exist

    # ques1_7
    def get_fasttext_matrix(self,  words_list, training_text ):
        # first sort words_list and then complete the function

        return M, word2Ind,  M_new, word2Ind_new    

    # ques2_a    docs is a list of documents.
    def co_occurrence(self,  docs,  k=2):
        # write your implementation to compute co-occurrence matrix M
        # and word to index dictionary word2Ind here
        return M, word2Ind

    # ques 2_b     M is co-occurrence matrix.
    def SVD_embedding(self,  M,  m=2):
        # write your implementation here
        return M_reduced

    # ques 2_c
    #input:
    #  M_reduced (numpy array of shape (V , k)): k-dimensional word embeddings
    #  word2Ind (dict): dictionary that maps word to indices for matrix M
    #  words (list of strings): words whose embeddings we want to visualize
    def plot_embeddings(self,  M_reduced, word2Ind, words):
        # write your implementation here
        # return statment is not required for this function
     


    