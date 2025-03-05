from gensim.models import KeyedVectors
import argparse
import torch

enRedDataset = "en.bin"
hiRedDataset = "hi.bin"
pairDataset  = "en-hi-cleaned.txt"
DIM = 300
BATCH = 128
DEVICE = "cuda"

parser = argparse.ArgumentParser(prog="supervised trainer")
parser.add_argument("--iter", type=int, default=0, help="Number of iterations to refine W by calculating nearest neighbour", required=False)
parser.add_argument("--dict", type=int, default=-1, help="Initial dictionary size",)
args = parser.parse_args()

ITER = args.iter
DICT = args.dict

en = KeyedVectors.load_word2vec_format(enRedDataset, binary=True)
hi = KeyedVectors.load_word2vec_format(hiRedDataset, binary=True)
enTen = torch.tensor(en.vectors, dtype=torch.float32, device=DEVICE)
hiTen = torch.tensor(hi.vectors, dtype=torch.float32, device=DEVICE)

def buildDict(w):
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
    
    mask = 0.001 < diff
    mask = mask.unsqueeze(1).expand_as(allPairs).clone()
    allPairs = allPairs.masked_select(mask).view(-1, 2)
    
    enEmb = enTen[allPairs[:, 0]]
    hiEmb = hiTen[allPairs[:, 1]]
    return enEmb, hiEmb

W = torch.nn.Linear(DIM, DIM, bias=False, device=DEVICE)

for iter in range(ITER):
    if iter == 0:
        #we used the known mapping in the first pass
        enWords, hiWords = [], []
        with open(pairDataset) as f:
            x = 0
            for line in f:
                if x == DICT: break
                enWord, hiWord = line.split()
                enWords.append(enWord)
                hiWords.append(hiWord)
                x += 1
        srcEmb = torch.tensor(en[enWords], requires_grad=False, device=DEVICE)
        tarEmb = torch.tensor(hi[hiWords], requires_grad=False, device=DEVICE)
        srcEmb = srcEmb / srcEmb.norm(p=2, dim=1, keepdim=True).expand_as(srcEmb)
        tarEmb = tarEmb / tarEmb.norm(p=2, dim=1, keepdim=True).expand_as(tarEmb)
    else:
        srcEmb, tarEmb = buildDict(W)
    M = tarEmb.T @ srcEmb
    U,S,V = torch.linalg.svd(M)
    with torch.no_grad(): W.weight.copy_(U @ V)
    diff = (W.weight@W.weight.T) - torch.eye(DIM, device=DEVICE)
    print(f"[{iter}]frobenius norm: {torch.norm(diff).detach().cpu().numpy()}")

torch.save(W, "w_sup.bin")
