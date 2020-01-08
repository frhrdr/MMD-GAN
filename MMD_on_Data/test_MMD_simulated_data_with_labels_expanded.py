""""
test a simple generating training using MMD for relatively simple datasets
with generating labels together with the input features
"""
# Mijung wrote on Dec 20, 2019

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import MMD_on_Data.util as util
import random


# generate data from 2D Gaussian for sanity check
def generate_data(mean_param, cov_param, n):

    how_many_gaussians = mean_param.shape[1]
    dim_gaussians = mean_param.shape[0]
    data_samps = np.zeros((n, dim_gaussians))
    labels = np.zeros((n, how_many_gaussians))

    samps_by_label = []
    for i in np.arange(0, how_many_gaussians):

        how_many_samps = np.int(n/how_many_gaussians)
        new_samps = np.random.multivariate_normal(mean_param[:, i], cov_param[:, :, i], how_many_samps)
        data_samps[(i*how_many_samps):((i+1)*how_many_samps), :] = new_samps
        labels[(i*how_many_samps):((i+1)*how_many_samps), i] = 1

        samps_by_label.append(new_samps)

    idx = np.random.permutation(n)
    shuffled_x = data_samps[idx, :]
    shuffled_y = labels[idx, :]

    return shuffled_x, shuffled_y, samps_by_label


def rff_gauss(n_features, x, w):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    w = torch.Tensor(w)
    xwt = torch.mm(x, torch.t(w))
    z1 = torch.cos(xwt)
    z2 = torch.sin(xwt)

    z = torch.cat((z1, z2), 1) * torch.sqrt(2.0/torch.Tensor([n_features]))
    return z


class GenerativeModel(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, n_classes):
            super(GenerativeModel, self).__init__()

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


def main(sample_target_embedding=False, learning_rate=1e-2, balanced_sampling=False):

    # random.seed(0)
    n = 6000  # number of data points divisible by num_Gassians
    num_gaussians = 3
    input_dim = 2
    mean_param = np.zeros((input_dim, num_gaussians))
    cov_param = np.zeros((input_dim, input_dim, num_gaussians))

    mean_param[:, 0] = [2, 8]
    mean_param[:, 1] = [-10, -4]
    mean_param[:, 2] = [-1, -7]

    cov_mat = np.empty((2,2))
    cov_mat[0, 0] = 1
    cov_mat[1, 1] = 4
    cov_mat[0, 1] = -0.25
    cov_mat[1, 0] = -0.25
    cov_param[:, :, 0] = cov_mat

    cov_mat[0, 1] = 0.4
    cov_mat[1, 0] = 0.4
    cov_param[:, :, 1] = cov_mat

    cov_param[:, :, 2] = 2 * np.eye(input_dim)

    data_samps, true_labels, samps_by_label = generate_data(mean_param, cov_param, n)

  # test how to use RFF for computing the kernel matrix
    med = util.meddistance(data_samps)
    # sigma2 = med**2
    sigma2 = med  # it seems to be more useful to use smaller length scale than median heuristic
    print('length scale from median heuristic is', sigma2)

    # random Fourier features
    n_features = 100
    n_classes = num_gaussians

    """ training a Generator via minimizing MMD """
    bs = 600
    if balanced_sampling:
        assert bs % n_classes == 0
        eye = np.eye(n_classes)
        balanced_label_batch = np.concatenate([np.stack([eye[i]]*(bs//n_classes)) for i in range(n_classes)])
        print(balanced_label_batch)
    else:
        balanced_label_batch = None

    input_size = 10
    hidden_size_1 = 100
    hidden_size_2 = 50
    output_size = input_dim + n_classes

    # model = Generative_Model(input_dim=input_dim, how_many_Gaussians=num_Gaussians)
    model = GenerativeModel(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                            output_size=output_size, n_classes=n_classes)

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    how_many_epochs = 1000
    how_many_iter = np.int(n/bs)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    draws = n_features // 2
    w_freq = np.random.randn(draws, input_dim) / np.sqrt(sigma2)

    """ computing mean embedding of true data """
    if sample_target_embedding is False:
        emb1_input_features = rff_gauss(n_features, torch.Tensor(data_samps), w_freq)  # (bs, rff)
        emb1_labels = torch.Tensor(true_labels)
        outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])  # ->(bs, rff, labels)
        mean_emb1 = torch.mean(outer_emb1, 0)
    else:
        mean_emb1 = None

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times
        batch_idx = 0
        running_loss = 0.0
        if sample_target_embedding is True:
            if balanced_sampling is True:
                samps_by_label = [s[np.random.permutation(s.shape[0]), :] for s in samps_by_label]
            else:
                idx = np.random.permutation(n)
                data_samps = data_samps[idx, :]
                true_labels = true_labels[idx, :]
        for i in range(how_many_iter):
            if sample_target_embedding is True:
                if balanced_sampling is True:
                    data_batch = np.concatenate([s[batch_idx*bs//n_classes:(batch_idx+1)*bs//n_classes, :]
                                                 for s in samps_by_label])
                    label_batch = balanced_label_batch
                else:
                    data_batch = data_samps[batch_idx*bs:(batch_idx+1)*bs, :]
                    label_batch = true_labels[batch_idx*bs:(batch_idx+1)*bs, :]
                emb1_input_features = rff_gauss(n_features, torch.Tensor(data_batch), w_freq)  # (bs, rff)
                emb1_labels = torch.Tensor(label_batch)
                outer_emb1 = torch.einsum('ki,kj->kij', [emb1_input_features, emb1_labels])  # ->(bs, rff, labels)
                mean_emb1 = torch.mean(outer_emb1, 0)
                batch_idx += 1

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(torch.randn((bs, input_size)))

            samp_input_features = outputs[:, 0:input_dim]
            samp_labels = outputs[:, -n_classes:]

            """ computing mean embedding of generated samples """
            emb2_input_features = rff_gauss(n_features, samp_input_features, w_freq)
            emb2_labels = samp_labels
            outer_emb2 = torch.einsum('ki,kj->kij', [emb2_input_features, emb2_labels])
            mean_emb2 = torch.mean(outer_emb2, 0)

            loss = torch.norm(mean_emb1-mean_emb2, p=2)**2

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if running_loss <= 1e-4:
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
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], c=labl, label=labl)
    plt.title('simulated data')

    plt.figure(2)
    plt.plot(training_loss_per_epoch)
    plt.title('MMD as a function of epoch')

    plt.show()


if __name__ == '__main__':
    main(sample_target_embedding=True, learning_rate=1e-2, balanced_sampling=False)
