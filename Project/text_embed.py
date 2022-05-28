def sen2index(sen,word2):
    new_sen=[word2[i] for i in sen]
    return new_sen

def word2index(word_lib):
    word2 = {tkn: i for i, tkn in enumerate(word_lib['word'], 1)}
    return word2

def idx2word(word2):
    index2 = {v: k for k, v in word2.items()}
    return index2