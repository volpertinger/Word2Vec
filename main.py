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

    # if normalized txt doesn't exist
    if not os.path.isfile(Settings.NORMALIZED_TEXT):
        Utils.write_valid_text(lemmas, Settings.NORMALIZED_TEXT)

    # if model doesn't exist
    if not os.path.isfile(Settings.MODEL):
        data = gensim.models.word2vec.LineSentence(Settings.NORMALIZED_TEXT)
        model = gensim.models.Word2Vec(data,
                                       window=Settings.WINDOW,
                                       min_count=Settings.MIN_COUNT,
                                       sg=Settings.SG,
                                       alpha=Settings.ALPHA)
        model.build_vocab(data)
        model.train(data, total_examples=model.corpus_count, epochs=Settings.EPOCHS, report_delay=Settings.REPORT_DELAY)
        model.save(Settings.MODEL)
    else:
        model = gensim.models.Word2Vec.load(Settings.MODEL)
    print(model.predict_output_word(["парень", "оружие"]))
    # print(model.wv.most_similar(positive=["сапог", "любовь"]))
