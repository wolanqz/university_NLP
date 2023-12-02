import gensim
import re

pos = ["оповещение_NOUN", "матрос_NOUN", "телеграмма_NOUN"]
neg = []
word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)
dist = word2vec.most_similar(positive=pos, negative=neg)

pat = re.compile("(.*)_NOUN")
for i in dist:
  e = pat.match(i[0])
  if e is not None:
    print(e.group(1))