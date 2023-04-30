import Settings
import Utils
import pickle
import os
import gensim
import numpy as np

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------------------------------------------------------
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

    # if normalized text doesn't exist
    if not os.path.isfile(Settings.NORMALIZED_TEXT):
        Utils.write_valid_text(lemmas, Settings.NORMALIZED_TEXT)

    # ------------------------------------------------------------------------------------------------------------------
    # Word2Vec model
    # ------------------------------------------------------------------------------------------------------------------

    # if w2v model doesn't exist
    if not os.path.isfile(Settings.MODEL_W2V):
        data = gensim.models.word2vec.LineSentence(Settings.NORMALIZED_TEXT)
        model_w2v = gensim.models.Word2Vec(data,
                                           window=Settings.WINDOW,
                                           min_count=Settings.MIN_COUNT,
                                           sg=Settings.SG,
                                           alpha=Settings.ALPHA,
                                           compute_loss=Settings.COMPUTE_LOSS,
                                           epochs=Settings.EPOCHS_W2V,
                                           vector_size=Settings.SIZE)
        model_w2v.build_vocab(data)
        model_w2v.train(data, total_examples=model_w2v.corpus_count, epochs=Settings.EPOCHS_W2V,
                        report_delay=Settings.REPORT_DELAY)
        model_w2v.save(Settings.MODEL_W2V)
    else:
        model_w2v = gensim.models.Word2Vec.load(Settings.MODEL_W2V)

    # ------------------------------------------------------------------------------------------------------------------
    # Plot examples
    # ------------------------------------------------------------------------------------------------------------------
    vectors = np.asarray(model_w2v.wv.vectors)
    labels = np.asarray(model_w2v.wv.index_to_key)
    x_vals, y_vals = Utils.reduce_dimensions(vectors)
    Utils.plot_with_matplotlib(x_vals, y_vals, labels, 34)
