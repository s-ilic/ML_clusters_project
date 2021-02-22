
import numpy as np
from tqdm import tqdm
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

fnames = [
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p001', #0
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-32_r0-32_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #1
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-128_r0-128_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #2
    'inn_mass_nmm-30_nperm-128_nshft-16_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #3
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_f1-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #4
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #5
    'inn_mass_nmm-10_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #6
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #7
    'inn_log10mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #8
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-none_dr-none_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001_8xpermvalid', #9
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-32_f1-32_f2-32_r0-32_r1-32_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #10
    'inn_mass_nmm-10_nperm-128_nshft-64_f0-32_f1-32_f2-32_r0-32_r1-32_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #11
    'inn_mass_nmm-5_nperm-128_nshft-64_f0-32_f1-32_f2-32_r0-32_r1-32_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #12
    'inn_mass_nmm-10_nperm-128_nshft-64_f0-64_f1-64_f2-64_r0-64_act-tanh_df-0p5_dr-0p5_l2f-0p001_l2r-0p001_loss-mean_squared_error_lr-0p0001', #13
    'inn_mass_nmm-10_nperm-128_nshft-64_f0-64_f1-64_f2-64_r0-64_act-tanh_df-0p5_dr-0p5_l2f-0p01_l2r-0p01_loss-mean_squared_error_lr-0p0001', #14
    'inn_mass_nmm-30_nperm-128_nshft-64_f0-64_r0-64_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p001', #15
    'inn_zspec_nmm-10_nperm-128_nshft-64_f0-128_f1-128_r0-128_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #16
    'inn_lambda_nmm-10_nperm-128_nshft-64_f0-128_f1-128_r0-128_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #17
    'inn_zspec_nmm-2_nperm-128_nshft-64_f0-128_f1-128_f2-128_act-tanh_df-0p5_dr-0p5_l2f-none_l2r-none_loss-mean_squared_error_lr-0p0001', #18
]
labs = [
    'Reference', #0
    '64 -> 32 neurons in f/rho', #1
    '64 -> 128 neurons in f/rho', #2
    '64 -> 16 random shifts', #3
    '1 -> 2 layers in f', #4
    'No dpout -> dpout 0.5', #5
    '30 -> 10 k-order in Janossy', #6
    '0.001 -> 0.0001 learning rate', #7
    'mass -> log10_mass target label', #8
    '8 x more perm for valid', #9
    'f=$32^3$, rho=$32^2$, dpout 0.5', #10
    'f=$32^3$, rho=$32^2$, dpout 0.5, k=10', #11
    'f=$32^3$, rho=$32^2$, dpout 0.5, k=5', #12
    'f=$64^3$, rho=$64$, dpout 0.5, k=10, l2reg=0.001 ', #13
    'f=$64^3$, rho=$64$, dpout 0.5, k=10, l2reg=0.01 ', #14
    'No dpout -> dpout 0.5, x10 learn. rate', #15
    'Redshift, f=$128^2$, rho=$128x1$, k=10, dpout=0.5', #16-z
    'Richness, f=$128^2$, rho=$128x1$, k=10, dpout=0.5', #17-l
    'f=$128^3$, rho=1, k=2, dpout=0.5', #18-z
]


if sys.argv[1] == '1':
    to_plot = [
        ## (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (7, 6),
        (7, 5),
        (7, 15),
    ]
    fname_out = 'tmp1'
elif sys.argv[1] == '2':
    to_plot = [
        ## (7, 7),
        ## (7, 8),
        ## (7, 9),
        (7, 10),
        (7, 11),
        (7, 12),
        (7, 13),
        (7, 14),
    ]
    fname_out = 'tmp2'
elif sys.argv[1] == '3':
    to_plot = [
        (16, 16),
        (17, 17),
        # (16, 18),
    ]
    fname_out = 'tmp3'
elif sys.argv[1] == '4':
    to_plot = [
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (7, 7),
        (8, 8),
        (7, 9),
        (7, 10),
        (7, 11),
        (7, 12),
        (7, 13),
        (7, 14),
        (7, 15),
    ]
    fname_out = 'tmp4'


cols = plt.cm.tab20(np.linspace(0., 1., 20, endpoint=False))

if len(sys.argv) == 2:
    to_plots = [to_plot]
else:
    to_plots = [[n] for n in to_plot]

for to_plot in to_plots:
    nr, nc = 1, 1
    while (nr*nc) < len(to_plot):
        if (1.*nc/nr) <= (16./9.):
            nc += 1
        else:
            nr += 1
    plt.figure(figsize=(16,8))
    ct = 1
    colix = 1
    for i, j in to_plot:
        plt.subplot(nr, nc, ct)
        ct += 1
        t = np.genfromtxt('saved_models/%s/output.log' % fnames[i], names=True)
        plt.plot(
            t['loss'],
            alpha=0.5,
            # label='loss ' + labs[0],
            label='Baseline loss',
            color=cols[0],
        )
        plt.plot(
            t['val_loss'],
            alpha=0.5,
            # label='val_loss ' + labs[0],
            label='Baseline val_loss',
            color=cols[1],
        )
        plt.axhline(t['val_loss'].min(), ls='--', color=cols[1])
        t = np.genfromtxt('saved_models/%s/output.log' % fnames[j], names=True)
        plt.plot(
            t['loss'],
            alpha=0.5,
            # label='loss ' + labs[i],
            color=cols[colix*2],
        )
        plt.plot(
            t['val_loss'],
            alpha=0.5,
            # label='val_loss ' + labs[i],
            color=cols[colix*2+1],
        )
        plt.axhline(t['val_loss'].min(), ls='--', color=cols[colix*2+1])
        plt.legend(loc='upper right')
        if sys.argv[1] != '3':
            if (i != 8) & (j != 8):
                plt.ylim(1., 4.)
            else:
                plt.yscale("log")
        else:
            plt.yscale("log")
        plt.title(labs[j])
        plt.xlabel("Epochs")
        # plt.xscale("log")
        colix += 1
    plt.tight_layout()
    if len(sys.argv) == 2:
        plt.savefig(fname_out+'.pdf')
    else:
        plt.savefig(fname_out+'_%s.pdf' % j)
    # plt.show()

'''
ssh n28
export CUDA_VISIBLE_DEVICES=""
cd ML
source to_load
cd ML_clusters_project/invariant_NN
python inn_mass_calc_nn.py
'''

'''
ssh n28
cd ML
source to_load
cd ML_clusters_project/invariant_NN
python inn_redshift_calc_nn.py
'''

'''
ssh n28
cd ML
source to_load
cd ML_clusters_project/invariant_NN
python inn_richness_calc_nn.py
'''