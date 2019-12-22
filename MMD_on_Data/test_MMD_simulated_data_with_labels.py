"""" test a simple generating training using MMD for relatively simple datasets """
""" with generating labels together with the input features """
# Mijung wrote on Dec 20, 2019

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random

# generate data from 2D Gaussian for sanity check
def generate_data(mean_param, cov_param, n):

    how_many_Gaussians = mean_param.shape[1]
    dim_Gaussians = mean_param.shape[0]
    data_samps = np.zeros((n, dim_Gaussians))
    labels = np.zeros((n, how_many_Gaussians))

    for i in np.arange(0,how_many_Gaussians):

        how_many_samps = np.int(n/how_many_Gaussians)
        new_samps = np.random.multivariate_normal(mean_param[:, i], cov_param[:, :, i], how_many_samps)
        data_samps[(i*how_many_samps):((i+1)*how_many_samps),:] = new_samps

        labels[(i*how_many_samps):((i+1)*how_many_samps),i] = 1


    idx = np.random.permutation(n)
    shuffled_x = data_samps[idx,:]
    shuffled_y = labels[idx,:]

    return shuffled_x, shuffled_y


def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    W = torch.Tensor(W)
    XWT = torch.mm(X, torch.t(W))
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]))
    return Z


class Generative_Model(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, n_classes):
            super(Generative_Model, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size
            self.n_classes = n_classes

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
            self.softmax = torch.nn.Softmax()

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.relu(output)
            output = self.fc3(output)

            output_features = output[:, 0:-self.n_classes]
            output_labels = output[:, -self.n_classes:]
            output_total = torch.cat((output_features, output_labels), 1)
            return output_total

def main():

    random.seed(0)

    n = 6000 # number of data points divisable by num_Gassians
    num_Gaussians = 3
    input_dim = 2
    mean_param = np.zeros((input_dim, num_Gaussians))
    cov_param = np.zeros((input_dim, input_dim, num_Gaussians))

    mean_param[:, 0] = [2, 8]
    mean_param[:, 1] = [-10, -4]
    mean_param[:, 2] = [-1, -7]

    cov_mat = np.empty((2,2))
    cov_mat[0,0] = 1
    cov_mat[1,1] = 4
    cov_mat[0,1] = -0.25
    cov_mat[1,0] = -0.25
    cov_param[:, :, 0] = cov_mat

    cov_mat[0,1] = 0.4
    cov_mat[1,0] = 0.4
    cov_param[:, :, 1] = cov_mat

    cov_param[:, :, 2] = 2 * np.eye(input_dim)

    data_samps, true_labels = generate_data(mean_param, cov_param, n)

  # test how to use RFF for computing the kernel matrix
    med = util.meddistance(data_samps)
    # sigma2 = med**2
    sigma2 = med # it seems to be more useful to use smaller length scale than median heuristic
    print('length scale from median heuristic is', sigma2)

    # random Fourier features
    n_features = 100
    n_classes = num_Gaussians

    """ training a Generator via minimizing MMD """
    mini_batch_size = 1000

    input_size = 10
    hidden_size_1 = 100
    hidden_size_2 = 50
    output_size = input_dim + n_classes

    # model = Generative_Model(input_dim=input_dim, how_many_Gaussians=num_Gaussians)
    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                             output_size=output_size, n_classes = n_classes)

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    how_many_epochs = 1000
    how_many_iter = np.int(n/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    draws = n_features // 2
    W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)

    """ computing mean embedding of true data """
    emb1_input_features = RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq)
    emb1_labels = torch.Tensor(true_labels)
    outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])
    mean_emb1 = torch.mean(outer_emb1, 0)

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(torch.randn((mini_batch_size, input_size)))

            samp_input_features = outputs[:,0:input_dim]
            samp_labels = outputs[:,-n_classes:]

            """ computing mean embedding of generated samples """
            emb2_input_features = RFF_Gauss(n_features, samp_input_features, W_freq)
            emb2_labels = samp_labels
            outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
            mean_emb2 = torch.mean(outer_emb2, 0)

            loss = torch.norm(mean_emb1-mean_emb2, p=2)**2

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if running_loss<=1e-4:
            break
        print('epoch # and running loss are ', [epoch, running_loss])
        training_loss_per_epoch[epoch] = running_loss

    plt.figure(1)
    plt.subplot(121)
    true_labl = np.argmax(true_labels, axis=1)
    plt.scatter(data_samps[:,0], data_samps[:,1], c=true_labl, label=true_labl)
    plt.title('true data')

    plt.subplot(122)
    model.eval()
    generated_samples = samp_input_features.detach().numpy()

    generated_labels = samp_labels.detach().numpy()
    labl = np.argmax(generated_labels, axis=1)
    plt.scatter(generated_samples[:,0], generated_samples[:,1], c=labl, label=labl)
    plt.title('simulated data')


    plt.figure(2)
    plt.plot(training_loss_per_epoch)
    plt.title('MMD as a function of epoch')

    plt.show()

if __name__ == '__main__':
    main()

