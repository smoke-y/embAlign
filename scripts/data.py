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
def cleanHiEn(src: str, dst: str) -> None:
    '''
    Read src and make sure all word pairs are
    present in our vocab. Else remove it
    '''
    en = KeyedVectors.load_word2vec_format(enRedDataset, binary=True)
    hi = KeyedVectors.load_word2vec_format(hiRedDataset, binary=True)
    file = open(src, "r", encoding="utf8")
    lines = []
    for line in file:
        enWord, hiWord = line.split()
        if hiWord not in hi or enWord not in en: continue
        lines.append(line)
    file.close()
    file = open(dst, "w+", encoding="utf8")
    file.writelines(lines)
    file.close()

#if processed dataset does not exist, process the dataset
if not path.exists(enRedDataset): reduceSave("wiki.en.vec", enRedDataset)
if not path.exists(hiRedDataset): reduceSave("wiki.hi.vec", hiRedDataset)
if not path.exists(pairDataset):  cleanHiEn("en-hi.5000.txt", pairDataset)
if not path.exists(pairTest):     cleanHiEn("en-hi.5000-6500.txt", pairTest)