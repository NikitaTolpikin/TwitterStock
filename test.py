from gensim.models import Word2Vec

model = Word2Vec.load('models/w2v/full/model_full.w2v')

word = ['мужик']

print(model.wv.most_similar(word, topn=6))
