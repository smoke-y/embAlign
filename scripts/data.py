from gensim.models import KeyedVectors
from os import path

enRedDataset = "en.bin"
hiRedDataset = "hi.bin"
pairDataset  = "en-hi-cleaned.txt"
pairTest     = "en-hi-cleaned-test.txt"
DIM = 300

def reduceSave(src: str, dst: str) -> None:
    '''
    Read src and pick the top 100,000 most frequent words
    and save it to dst
    '''
    model = KeyedVectors.load_word2vec_format(src)
    mostCom = list(model.key_to_index.keys())[:100000]
    reducedModel = KeyedVectors(vector_size=model.vector_size)
    assert DIM == model.vector_size, f"Expected vector dimension is not same as dataset's dim. {DIM} != {model.vector_size}"
    reducedModel.add_vectors(mostCom, [model[word] for word in mostCom])
    reducedModel.save_word2vec_format(dst, binary=True)

#if processed dataset does not exist, process the dataset
if not path.exists(enRedDataset): reduceSave("wiki.en.vec", enRedDataset)
if not path.exists(hiRedDataset): reduceSave("wiki.hi.vec", hiRedDataset)
if not path.exists(pairDataset) or not path.exists(pairTest):
    en = KeyedVectors.load_word2vec_format(enRedDataset, binary=True)
    hi = KeyedVectors.load_word2vec_format(hiRedDataset, binary=True)
    file = open("en-hi.txt", "r", encoding="utf8")
    lines = []
    #go through our dataset and make sure the pair is present in our vocab
    for line in file:
        enWord, hiWord = line.split()
        if hiWord not in hi or enWord not in en: continue
        lines.append(line)
    file.close()
    #split into test and train dataset
    trainLen = len(lines) - 2000
    file = open(pairDataset, "w+", encoding="utf8")
    file.writelines(lines[:trainLen])
    file.close()
    file = open(pairTest, "w+", encoding="utf8")
    file.writelines(lines[trainLen:])
    file.close()