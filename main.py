import Settings
import Utils
import pickle
import os
import gensim

if __name__ == '__main__':
    input_text = open(Settings.PLAIN_INPUT, "r", encoding=Settings.PLAIN_ENCODING)

    # get lemmas - from file or generating if file doesn't exist
    if os.path.isfile(Settings.LEMMAS):
        file = open(Settings.LEMMAS, 'rb')
        lemmas = pickle.load(file)
        file.close()
    else:
        file = open(Settings.LEMMAS, 'wb')
        lemmas = Utils.lemmatize(input_text.readlines())
        pickle.dump(lemmas, file)
        file.close()
    print(lemmas)

    # if normalized txt doesn't exist
    if not os.path.isfile(Settings.NORMALIZED_TEXT):
        Utils.write_valid_text(lemmas, Settings.NORMALIZED_TEXT)

    data = gensim.models.word2vec.LineSentence(Settings.NORMALIZED_TEXT)
    model = gensim.models.Word2Vec(data, window=10, min_count=2, sg=0)
    model.init_sims(replace=True)
    print(len(model.wv.vocab))
    model.save('my.model')
