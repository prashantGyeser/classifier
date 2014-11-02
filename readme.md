
Read me:

Use example:

import classyFier as classyFier
from classyFier import StemmedTfidfVectorizer
classy= classyFier.loadClassifierVectorizerComponents('./trainedclassifiervectorizer.pkl')
classy(["this is a cup", "  i want food", "ET go home"])
