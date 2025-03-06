from gensim.models import KeyedVectors
from sys import argv
import torch

assert len(argv) == 2, "Please provide the path for the W matrix"

enRedDataset = "en.bin"
hiRedDataset = "hi.bin"
pairDataset  = "en-hi-cleaned-test.txt"
DIM = 300
DEVICE = "cuda"

en = KeyedVectors.load_word2vec_format(enRedDataset, binary=True)
hi = KeyedVectors.load_word2vec_format(hiRedDataset, binary=True)
enTen = torch.tensor(en.vectors, dtype=torch.float32, device=DEVICE)
hiTen = torch.tensor(hi.vectors, dtype=torch.float32, device=DEVICE)
hiTen = hiTen / hiTen.norm(p=2, dim=1, keepdim=True).expand_as(hiTen)
W = torch.load(argv[1], weights_only=False).to(DEVICE)

f = open(pairDataset, "r", encoding="utf8")
p1 = 0
p5 = 0
x = 0
greaterThan0 = 0
lessThan0 = 0
strongCor = 0
modCor = 0
strongCorPairs = []
for line in f:
    x += 1
    enWord, hiWord = line.split()
    srcEmb = W(torch.tensor(en[enWord], device=DEVICE, requires_grad=False).unsqueeze(0))
    tarEmb = torch.tensor(hi[hiWord], device=DEVICE, requires_grad=False).unsqueeze(0)
    srcEmb = srcEmb / srcEmb.norm(p=2, dim=1, keepdim=True).expand_as(srcEmb)
    tarEmb = tarEmb / tarEmb.norm(p=2, dim=1, keepdim=True).expand_as(tarEmb)

    #analyzing srcEmb and tarEmb by taking dot prod
    theta = srcEmb @ tarEmb.T
    theta = theta.detach().cpu().numpy()[0]
    if theta > 0.0: greaterThan0 += 1
    else: lessThan0 += 1
    if theta > 0.75:
        strongCor += 1
        strongCorPairs.append([enWord, hiWord])
    elif theta > 0.5: modCor += 1

    #p@1 and p@5 by taking dot prod
    tarIndex = hi.key_to_index[hiWord]
    scores = hiTen.mm(srcEmb.transpose(0, 1)).transpose(0, 1)
    topScores, topTargets = scores.topk(5, dim=1)
    topTargets = topTargets.detach().squeeze(0).cpu().numpy()
    topScores = topScores.detach().squeeze(0).cpu().numpy()
    if topTargets[0] == tarIndex: p1 += 1
    if tarIndex in topTargets: p5 += 1

print("total pairs processed: ", x)
print(f"precision@1: {(p1/x)*100}%\nprecision@5: {(p5/x)*100}%")
print(f"[cosine similarity]\ngreater than 0: {greaterThan0}\nless than 0: {lessThan0}")
print(f"strong corelation: {strongCor}\nmoderate corelation: {modCor}\nweak corelation: {x-strongCor+modCor}")
print("[strong corelation words]")
for i in strongCorPairs: print(i[0], i[1])
f.close()
