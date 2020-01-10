# This is for computing the cumulative privacy loss of our algorithm
# We use the analytic moments accountant method by Wang et al
# (their github repo is : https://github.com/yuxiangw/autodp)
# by changing the form of upper bound on the Renyi DP, resulting from
# several Gaussian mechanisms we use given a mini-batch.
from autodp import rdp_acct, rdp_bank

# get the CGF functions
def CGF_func(sigma1, sigma2, sigma3, sigma4, num_Clust, num_iter_EM):

    # gaussian 1 and 2 are for the discrimintor update (i.e., two terms for applying DP-SGD)
    func_gaussian_1 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma1}, x)
    func_gaussian_2 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma2}, x)

    # gaussian 3 and 4 are for EM updates for MoG
    func_gaussian_3 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma3}, x)
    func_gaussian_4 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma4}, x)

    func = lambda x: func_gaussian_1(x) + func_gaussian_2(x) + num_Clust*num_iter_EM*(func_gaussian_3(x) + func_gaussian_4(x))
    return func


def main():

    """ input arguments """

    # (1) privacy parameters for four types of Gaussian mechanisms
    sigma1 = 2.
    sigma2 = 200.0
    sigma3 = 200.0
    sigma4 = 200.0

    # (2) number of clusters in MoG
    num_Clust = 1

    # (3) number of iterations in EM updates
    num_iter_EM = 1

    # (4) desired delta level
    delta = 1e-5

    # (5) number of training steps
    k = 4000

    # (6) sampling rate
    prob = 512./60000.

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    # define the functional form of uppder bound of RDP
    func = CGF_func(sigma1, sigma2, sigma3, sigma4, num_Clust, num_iter_EM)

    eps_seq = []
    print_every_n = 100
    for i in range(1, k+1):
        acct.compose_subsampled_mechanism(func, prob)
        eps_seq.append(acct.get_eps(delta))
        if i % print_every_n == 0 or i == k:
            print("[", i, "]Privacy loss is", (eps_seq[-1]))

    print("Composition of 1000 subsampled Gaussian mechanisms gives ", (acct.get_eps(delta), delta))


if __name__ == '__main__':
    main()
