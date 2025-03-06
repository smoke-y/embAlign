from gensim.models import KeyedVectors
from torch.autograd import Variable
import torch.nn as nn
import torch

enRedDataset = "en.bin"
hiRedDataset = "hi.bin"
#Good idea to put it under a class but it's only 5
DIM = 300
DEVICE = "cuda"
#From section 3.1
DECAY = 0.0005
BATCH = 32
LR = 0.1
EPOCH = 5
DIS_STEPS = 5
EPOCH_SIZE = 1000000

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

def buildDict(w, enTen, hiTen):
    srcEmb = w(enTen)
    tarEmb = hiTen
    srcEmb = srcEmb / srcEmb.norm(p=2, dim=1, keepdim=True).expand_as(srcEmb)
    tarEmb = tarEmb / tarEmb.norm(p=2, dim=1, keepdim=True).expand_as(tarEmb)
    nSrc = srcEmb.size(0)
    allScores, allTargets = [], []
    #nearest neighbour
    for i in range(0, nSrc, BATCH):
        scores = tarEmb.mm(srcEmb[i:min(nSrc, i+BATCH)].transpose(0, 1)).transpose(0, 1)
        bestScores, bestTarget = scores.topk(2, dim=1)
        allScores.append(bestScores)
        allTargets.append(bestTarget)
    allScores = torch.cat(allScores)
    allTargets = torch.cat(allTargets)

    allPairs = torch.cat([
        torch.arange(0, allTargets.size(0), device=DEVICE).long().unsqueeze(1),
        allTargets[:, 0].unsqueeze(1)
    ], dim=1)

    diff = allScores[:, 0] - allScores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    allScores = allScores[reordered]
    allPairs = allPairs[reordered]
    
    mask = 0.01 < diff
    mask = mask.unsqueeze(1).expand_as(allPairs).clone()
    allPairs = allPairs.masked_select(mask).view(-1, 2)
    
    enEmb = enTen[allPairs[:, 0]]
    hiEmb = hiTen[allPairs[:, 1]]
    return enEmb, hiEmb

def train(D: Discriminator, M: Mapping, modelPath: str) -> None:
    en = KeyedVectors.load_word2vec_format(enRedDataset, binary=True)
    hi = KeyedVectors.load_word2vec_format(hiRedDataset, binary=True)
    en = torch.tensor(en.vectors, dtype=torch.float32, device=DEVICE)
    hi = torch.tensor(hi.vectors, dtype=torch.float32, device=DEVICE)
    dOptim = torch.optim.SGD(D.parameters(), LR, weight_decay=DECAY)
    mOptim = torch.optim.SGD(M.parameters(), LR, weight_decay=DECAY)
    y = Variable(torch.FloatTensor(2 * 32)).to(DEVICE)
    y[:BATCH] = 0
    y[BATCH:] = 1
    #adversial training
    for e in range(EPOCH):
        for i in range(0, EPOCH_SIZE, BATCH):
            for j in range(DIS_STEPS):
                srcIds = torch.LongTensor(BATCH).random_(len(en)).to(DEVICE)
                tarIds = torch.LongTensor(BATCH).random_(len(hi)).to(DEVICE)
                srcEmb = en[srcIds]
                tarEmb = hi[tarIds]
                srcEmb = srcEmb / srcEmb.norm(p=2, dim=1, keepdim=True).expand_as(srcEmb)
                tarEmb = tarEmb / tarEmb.norm(p=2, dim=1, keepdim=True).expand_as(tarEmb)

                pred = M(srcEmb)
                x = torch.cat([pred, tarEmb])

                #discriminator weight update
                dPred = D(x.detach())
                dLoss = torch.nn.functional.binary_cross_entropy(dPred.view(-1), y)
                dOptim.zero_grad()
                dLoss.backward()
                dOptim.step()
            srcIds = torch.LongTensor(BATCH).random_(len(en)).to(DEVICE)
            tarIds = torch.LongTensor(BATCH).random_(len(hi)).to(DEVICE)
            srcEmb = en[srcIds]
            tarEmb = hi[tarIds]
            srcEmb = srcEmb / srcEmb.norm(p=2, dim=1, keepdim=True).expand_as(srcEmb)
            tarEmb = tarEmb / tarEmb.norm(p=2, dim=1, keepdim=True).expand_as(tarEmb)

            pred = M(srcEmb)
            x = torch.cat([pred, tarEmb])

            # #mapping weight update
            dPred = D(x)
            mLoss = torch.nn.functional.binary_cross_entropy(dPred.view(-1), 1-y)
            mOptim.zero_grad()
            mLoss.backward()
            mOptim.step()
            M.orthogonalize()

    print(dLoss.detach().cpu().numpy(), dLoss.detach().cpu().numpy())
    #procrustes algorithm
    for i in range(3):
        srcEmb, tarEmb = buildDict(M.W, en, hi)
        Mm = tarEmb.T @ srcEmb
        U,S,V = torch.linalg.svd(Mm)
        with torch.no_grad(): M.W.weight.copy_(U @ V)
    torch.save(M.W, modelPath)

train(Discriminator().to(DEVICE), Mapping().to(DEVICE), "w_unsup.bin")