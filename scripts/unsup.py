from gensim.models import KeyedVectors
from torch.autograd import Variable
from itertools import islice
import torch.nn as nn
import numpy as np
import torch

enRedDataset = "en.bin"
hiRedDataset = "hi.bin"
pairDataset  = "en-hi-cleaned.txt"
#Good idea to put it under a class but it's only 4
DIM = 300
#From section 3.1
DECAY = 0.95
BATCH = 32
LR = 0.1

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        #Implemented according to section 3.1
        self.model = nn.Sequential(
            nn.Linear(DIM, 2048),
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
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(DIM, DIM, bias=False)
        self.beta = 0.01
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.W(x)
    def orthogonalize(self) -> None:
        with torch.no_grad():
            W = self.W.weight
            self.W.weight.copy_((1 + self.beta) * W - self.beta * W.mm(W.transpose(0, 1).mm(W)))

def train(D: Discriminator, M: Mapping, epoch: int, modelPath: str) -> None:
    file = open(pairDataset, "r", encoding="utf8")
    en = KeyedVectors.load_word2vec_format(enRedDataset, binary=True)
    hi = KeyedVectors.load_word2vec_format(hiRedDataset, binary=True)
    dOptim = torch.optim.SGD(D.parameters(), LR, weight_decay=DECAY)
    mOptim = torch.optim.SGD(M.parameters(), LR, weight_decay=DECAY)
    y = Variable(torch.FloatTensor(2 * 32))
    y[:BATCH] = 0
    y[BATCH:] = 1
    for e in range(epoch):
        file.seek(0)
        while True:
            lines = list(islice(file, BATCH))
            if len(lines) < BATCH: break
            hiList, enList = [], []
            for line in lines:
                enWord, hiWord = line.split()
                hiList.append(hi[hiWord])
                enList.append(en[enWord])
            hiEmb, enEmb = torch.Tensor(np.array(hiList)), torch.Tensor(np.array(enList))

            pred = M(enEmb)
            x = torch.cat([pred, hiEmb])

            #discriminator weight update
            dPred = D(x.detach())
            dLoss = torch.nn.functional.binary_cross_entropy(dPred.view(-1), y)
            dOptim.zero_grad()
            dLoss.backward()
            dOptim.step()

            # #mapping weight update
            dPred = D(x)
            mLoss = torch.nn.functional.binary_cross_entropy(dPred.view(-1), 1-y)
            mOptim.zero_grad()
            mLoss.backward()
            mOptim.step()
            M.orthogonalize()

            print(dLoss.detach().numpy(), dLoss.detach().numpy())
    file.close()
    torch.save(M.W, modelPath)

train(Discriminator(), Mapping(), 5, "w_unsup.bin")