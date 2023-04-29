import Settings
import Utils
import pickle
import os
import gensim

if __name__ == '__main__':
    input_text = open("O Henry. Russkie sobolya.txt", "r", encoding="UTF-8")

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

    valid_text = Utils.get_valid_text(lemmas)
    print(valid_text)

    data = gensim.models.word2vec.LineSentence(valid_text)
    model = gensim.models.Word2Vec(data, Settings.SIZE, Settings.WINDOW, Settings.MIN_COUNT, Settings.SG)
    model.init_sims(replace=True)
    print(len(model.wv.vocab))
    model.save(Settings.MODEL)
