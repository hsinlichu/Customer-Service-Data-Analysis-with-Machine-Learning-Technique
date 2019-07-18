import torch

class Metric():
    def __init__(self):
        self.n = 0
        self.n_correct = 0

    def update(self, output, gt):
        maxindex = torch.argmax(output, dim=1)
        #pdb.set_trace()
        for i in range(len(gt)):
            self.n += 1
            if gt[i][maxindex[i]] == 1:
                self.n_correct += 1

    def reset(self):
        self.n = 0
        self.n_correct = 0

    def get_score(self):
        score = self.n_correct/self.n
        return "{:.3f}".format(score)
        pass


