import Settings
import Utils
import pickle
import os

if __name__ == '__main__':
    input_text = open("O Henry. Russkie sobolya.txt", "r", encoding="UTF-8")

    # get lemmas - from file or generating if file doesn't exists
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
