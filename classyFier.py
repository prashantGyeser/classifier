import numpy as np
import nltk as nltk
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 


from sklearn.externals import joblib
eng_stemmer = SnowballStemmer('english')

"""
use case:  
import classyFier as classyFier
from classyFier import StemmedTfidfVectorizer
classy= classyFier.loadClassifierVectorizerComponents('./trainedclassifiervectorizer.pkl')
classy(["this is a cup", "  i want food", "ET go home"])

Output: array([ 0.,  1., 0.])

"""

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (eng_stemmer.stem(w) for w in analyzer(doc))


def loadClassifierVectorizerComponents(fileName ='./trainedclassifiervectorizer.pkl'):
    clf,vect =joblib.load(fileName)
    f1 = lambda x1: clf.predict(vect.transform(x1))
    return f1

if __name__ == '__main__':
    print "use case:  classy = classyFier.loadClassifierVectorizerComponents('./trainedclassifiervectorizer.pkl')"
