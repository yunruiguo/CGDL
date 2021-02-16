import os
import matplotlib.pyplot as plt
os.environ['R_HOME'] = "/home/guo/.conda/envs/torch/lib/R" #or whereever your R is installed
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_curve, auc
import argparse
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
numpy2ri.activate()
mvt = importr('mvtnorm')
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='PyTorch OSR Example')
parser.add_argument('--lamda', type=int, default=100, help='lamda in loss function')
parser.add_argument('--num_class', type=int, default=6, help='number of class')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold of gaussian model')
args = parser.parse_args()
def revise(epoch):
    train_rec = np.loadtxt('lvae%d/train_rec.txt' %args.lamda)
    rec_mean = np.mean(train_rec)
    rec_std = np.std(train_rec)
    rec_thres = rec_mean + 2 * rec_std #95%

    omn_rec = np.loadtxt('lvae%d/omn_rec.txt' %args.lamda)
    omn_pre = np.loadtxt('lvae%d/omn_pre.txt' %args.lamda)
    omn_pre[(omn_rec > rec_thres)] = args.num_class
    open('lvae%d/omn_pre.txt' %args.lamda , 'w').close()  # clear
    np.savetxt('lvae%d/omn_pre.txt' %args.lamda , omn_pre, delimiter=' ', fmt='%d')


class GAU(object):
    def __init__(self, file_path):
        self.trainfea = np.loadtxt(os.path.join(file_path, 'train_fea.txt'))
        self.traintar = np.loadtxt(os.path.join(file_path, 'train_tar.txt'))
        self.labelset = set(self.traintar)
        self.labelnum = len(self.labelset)
        self.num = np.shape(self.trainfea)[0]
        self.dim = np.shape(self.trainfea)[1]
        self.gau = self.train()

    def train(self):
        trainfea = self.trainfea
        traintar = self.traintar
        labelnum = self.labelnum
        trainsize = self.trainfea.shape[0]
        for i in range(labelnum):
            locals()['matrix' + str(i)] = np.empty(shape=[0,self.dim])

        gau = []
        muandsigma = []
        for j in range(trainsize):
            for i in range(labelnum):
                if traintar[j] == i:
                    locals()['matrix' + str(i)] = np.append((locals()['matrix' + str(i)]), np.array([np.array(trainfea[j])]),
                                                            axis=0)

        for i in range(labelnum):
            locals()['mu' + str(i)] = np.mean(np.array(locals()['matrix' + str(i)]),axis=0)
            locals()['sigma' + str(i)] = np.cov(np.array((locals()['matrix' + str(i)] - locals()['mu' + str(i)])).T)
            locals()['gau' + str(i)] = [locals()['mu' + str(i)],locals()['sigma' + str(i)]]
            print(i)
            print(locals()['mu' + str(i)])
            print(np.diag(locals()['sigma' + str(i)])**0.5)
            gau.append(locals()['gau' + str(i)])

        return gau

    def test(self,testsetlist,threshold = args.threshold):

        testfea = np.loadtxt(testsetlist[0])
        testtar = np.loadtxt(testsetlist[1])
        testpre = np.loadtxt(testsetlist[2])

        labelnum = self.labelnum
        gau = self.gau
        dim = self.dim
        performance = np.zeros([labelnum + 1, 5])
        testsize = testfea.shape[0]
        result = []
        if threshold != 0:
            print('threshold is', threshold)

        def multivariateGaussian(vector, mu, sigma):
            vector = np.array(vector)
            d = (np.mat(vector - mu)) * np.mat(np.linalg.pinv(sigma)) * (np.mat(vector - mu).T)
            p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
            p = float(p)
            return p

        def multivariateGaussianNsigma(sigma,threshold):
            q = np.array(mvt.qmvnorm(threshold, sigma = sigma, tail = "both")[0])
            n = q[0]
            m = (np.diag(sigma) ** 0.5) * n
            d = (np.mat(m) * np.mat(np.linalg.pinv(sigma)) * (np.mat(m).T))
            p = np.exp(d * (-0.5)) / (((2 * np.pi) ** int(dim/2)) * (np.linalg.det(sigma)) ** (0.5))
            return p

        pNsigma = np.zeros(labelnum)
        p = np.zeros(labelnum)
        mu = []
        sigma = []
        P = []
        Y = []
        for j in range(labelnum):
            mu.append(gau[j][0])
            sigma.append(gau[j][1])
            pNsigma[j] = multivariateGaussianNsigma(sigma[j], threshold)

        with tqdm(total=testsize) as pbar:
            for i in range(testsize):
                for j in range(labelnum):
                    p[j] = multivariateGaussian(testfea[i],mu[j],sigma[j])

                delta = p#-pNsigma
                P += [np.max(delta)]
                if int(testtar[i]) >= labelnum:
                    testtar[i] = labelnum
                    Y += [0]
                else:
                    Y += [1]
                # print(delta)
                if len(delta[delta > 0]) == 0:
                    #Unseen
                    prediction = labelnum
                else:
                    #Seen
                    prediction = testpre[i]

                result.append(prediction)
                pbar.update(i)
        P = np.asarray(P)
        Y = np.asarray(Y)

        fpr, tpr, _ = roc_curve(Y, P, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print("AUC: ", roc_auc)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        #result
        result = np.array(result)

        for i in range(labelnum+1):
            locals()['resultIndex' + str(i)] = np.argwhere(result == i)
            locals()['targetIndex' + str(i)] = np.argwhere(testtar == i)

        for i in range(labelnum+1):
            locals()['tp' + str(i)] = np.sum((testtar[(locals()['resultIndex' + str(i)])]) == i)
            locals()['fp' + str(i)] = np.sum((testtar[(locals()['resultIndex' + str(i)])]) != i)
            locals()['fn' + str(i)] = np.sum((result[locals()['targetIndex' + str(i)]]) != i)
            # print(locals()['tp' + str(i)],locals()['fp' + str(i)],locals()['fn' + str(i)])

            performance[i, 0] = locals()['tp' + str(i)]
            performance[i, 1] = locals()['fp' + str(i)]
            performance[i, 2] = locals()['fn' + str(i)]

        for i in range(labelnum+1):
            locals()['precision' + str(i)] = locals()['tp' + str(i)]/(locals()['tp' + str(i)]+locals()['fp' + str(i)] + 1)
            locals()['recall' + str(i)] = locals()['tp' + str(i)]/(locals()['tp' + str(i)]+locals()['fn' + str(i)] + 1)
            locals()['f-measure' + str(i)] = 2* locals()['precision' + str(i)]*locals()['recall' + str(i)]/(locals()['precision' + str(i)] + locals()['recall' + str(i)])

            performance[i, 3] = locals()['precision' + str(i)]
            performance[i, 4] = locals()['recall' + str(i)]

        performancesum = performance.sum(axis = 0)
        mafmeasure = 2*performancesum[3]*performancesum[4]/(performancesum[3]+performancesum[4])
        maperformance = np.append((performancesum)[3:],mafmeasure)/(labelnum+1)

        print(performance)
        np.savetxt('lvae%d/performance.txt' %args.lamda , performance)
        return maperformance

if __name__ == '__main__':
    parser.add_argument('--dataset', default='mnist', help='cifar10|cifar100')
    parser.add_argument('--split', default='split1', help='split0, split1, ...')
    parser.add_argument('--results_dir', default='./results', help='./results')
    args = parser.parse_args()
    split_results_dir = os.path.join(args.results_dir, args.dataset, args.split)
    for epoch in range(1):

        #revise(epoch)
        gau = GAU(split_results_dir)

        omn = [os.path.join(split_results_dir, 'omn_fea.txt'),
               os.path.join(split_results_dir, 'omn_tar.txt'),
               os.path.join(split_results_dir, 'omn_pre.txt')]

        perf_omn = gau.test(omn, args.threshold)
        #
        ma = [perf_omn]
        print(ma)
        np.savetxt(os.apth.join(split_results_dir, 'ma.txt'), ma)