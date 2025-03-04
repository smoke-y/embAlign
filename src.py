from gensim.models import KeyedVectors
import torch.nn as nn
from os import path
import torch

enRedDataset = "en.bin"
hiRedDataset = "hi.bin"

def reduceSave(src: str, dst: str) -> None:
    '''
    Read src and pick the top 100,000 most frequent words
    and save it to dst
    '''
    model = KeyedVectors.load_word2vec_format(src)
    mostCom = list(model.key_to_index.keys())[:100000]
    reducedModel = KeyedVectors(vector_size=model.vector_size)
    print(f"{src} vector dim is {model.vector_size}")
    reducedModel.add_vectors(mostCom, [model[word] for word in mostCom])
    reducedModel.save_word2vec_format(dst, binary=True)

#if processed dataset does not exist, process the dataset
if not path.exists(enRedDataset): reduceSave("wiki.en.vec", enRedDataset)
if not path.exists(hiRedDataset): reduceSave("wiki.hi.vec", hiRedDataset)

class Discriminator(nn.Module):
    def __init__(self, embDim: int):
        super().__init__()
        #Implemented according to section 3.1
        self.model = nn.Sequential(
            nn.Linear(embDim, 2048),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.model(x)

class Mapping(nn.Module):
    def __init__(self, embDim: int):
        super().__init__()
        self.W = nn.Linear(embDim, embDim, bias=False)
        self.beta = 0.01
    def makeOrthogonal(self, w: torch.Tensor) -> torch.Tensor:
        #Implemented according to section 3.3
        with torch.no_grad(): return (1+self.beta)*w - self.beta*((w @ w.T) @ w)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.makeOrthogonal(self.W(x))