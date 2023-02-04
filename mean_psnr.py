results_root = '/home/loyot/workspace/code/training_results/nerfacc'
model = 'ngp_dnerf'
scenes=["bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex"]

feats = ['no_pf', 'pf', 'pf_mix']
lrs = ['lr_0-003', 'lr_0-01']

# scenes_lr = {
#     "bouncingballs":1e-2, 
#     "hellwarrior":1e-2, 
#     "hook": 1e-2, 
#     "jumpingjacks": 1e-2, 
#     "lego": 1e-2, 
#     "mutant": 1e-2, 
#     "standup": 5e-3, 
#     "trex": 5e-3}
# for lr in lrs[-1:]:
#     for feat in feats:
#         psnrs = []
#         for scene in scenes:
#             if scene == 'bouncingballs':
#                 scene_dir = f'{results_root}/{model}/{scene}/{lrs[0]}/{feat}/psnr.txt'
#             else:
#                 scene_dir = f'{results_root}/{model}/{scene}/{lr}/{feat}/psnr.txt'
#             with open(scene_dir, 'r') as f:
#                 data = f.readlines()
#                 for d in data:
#                     if 'mean' in d:
#                         psnrs.append(float(d.split(' ')[-1]))
#         avg = sum(psnrs) / len(psnrs)
#         cat_psnrs = [f'{scenes[i]}: {psnrs[i]:.5f}, 'for i in range(len(scenes))]

#         cat_psnrs_str = ''
#         for p in cat_psnrs:
#             cat_psnrs_str+=p
#         print(f'{cat_psnrs_str}')

#         print(f'{results_root}/{model}/{scene}/{lr}/{feat}-avg psnr: {avg}')


# lrs = [0.01, 0.015, 0.005, 0.003, 0.002, 0.001, 0.0005]
# for lr_n in lrs:
#     lr = 'lr_' + str(lr_n).replace('.', '-')
#     print(f"lr: {lr_n}")
#     feat = 'pf_nopw_l-huber_dive'
#     psnrs = []
#     for scene in scenes:
#         scene_dir = f'{results_root}/{model}/{scene}/{lr}/{feat}/psnr.txt'
#         with open(scene_dir, 'r') as f:
#             data = f.readlines()
#             for d in data:
#                 if 'mean' in d:
#                     psnrs.append(float(d.split(' ')[-1]))
#     avg = sum(psnrs) / len(psnrs)
#     cat_psnrs = [f'{scenes[i]}: {psnrs[i]:.5f}, 'for i in range(len(scenes))]

#     cat_psnrs_str = ''
#     for p in cat_psnrs:
#         cat_psnrs_str+=p
#     print(f'{cat_psnrs_str}')

#     print(f'{results_root}/{model}/{scene}/{lr}/{feat}-avg psnr: {avg}')

# lrs = [0.01, 0.015, 0.005, 0.003, 0.002, 0.001, 0.0005]
# for lr_n in lrs:
#     lr = 'lr_' + str(lr_n).replace('.', '-')
#     print(f"lr: {lr_n}")
#     feat = 'pf_nopw_l-huber_dive'
#     psnrs = []
#     for scene in scenes:
#         scene_dir = f'{results_root}/{model}/{scene}/{lr}/{feat}/psnr.txt'
#         with open(scene_dir, 'r') as f:
#             data = f.readlines()
#             for d in data:
#                 if 'mean' in d:
#                     psnrs.append(float(d.split(' ')[-1]))
#     avg = sum(psnrs) / len(psnrs)
#     cat_psnrs = [f'{scenes[i]}: {psnrs[i]:.5f}, 'for i in range(len(scenes))]

#     cat_psnrs_str = ''
#     for p in cat_psnrs:
#         cat_psnrs_str+=p
#     print(f'{cat_psnrs_str}')

#     print(f'{results_root}/{model}/{scene}/{lr}/{feat}-avg psnr: {avg}')

lrs = [0.01,]
# feats of diff feat
# feats = [
#     'pf_nopw_l-huber_te',
#     'pf_pw_l-huber_op',
#     'pf_nopw_l-huber',
#     'pf_pw_l-huber_op_te',
#     'nopf_pw_l-huber_te_ta',
#     'nopf_nopw_l-huber',
#     'nopf_pw_l-huber',
#     'nopf_nopw_l-huber_te_ta',
#     'pf_nopw_l-huber_op_te_ta',
#     'pf_pw_l-huber_op_te_ta',
#     'nopf_pw_l-huber_op',
#     'pf_pw_l-huber',
#     'nopf_pw_l-huber_op_te',
#     'pf_nopw_l-huber_dive_op_te_ta',
#     'pf_nopw_l-huber_op_te',
#     'pf_nopw_l-huber_op',
#     'nopf_pw_l-huber_te',
#     'pf_pw_l-huber_te',
#     'pf_nopw_l-huber_te_ta',
#     'nopf_nopw_l-huber_op_te',
#     'nopf_pw_l-huber_op_te_ta',
#     'pf_pw_l-huber_te_ta',
#     'nopf_nopw_l-huber_op_te_ta',
#     'nopf_nopw_l-huber_te'
# ]



# feats of diff dive
feats = [
    'hl2_pf_nopw_l-huber_dive64.0_op_te_ta',
    'hl2_pf_nopw_l-huber_dive128.0_op_te_ta',
    'hl2_pf_nopw_l-huber_dive256.0_op_te_ta',
    'hl2_pf_nopw_l-huber_dive512.0_op_te_ta',
    'hl2_pf_nopw_l-huber_dive1024.0_op_te_ta',
    'hl2_pf_nopw_l-huber_dive2048.0_op_te_ta',
    'hl2_pf_nopw_l-huber_dive4096.0_op_te_ta',
    'hl2_pf_nopw_l-huber_dive8192.0_op_te_ta'
]

for lr_n in lrs:
    # feats = ['nopf_nopw_l-huber', 'nopf_nopw_l-huber_op', 'pf_nopw_l-huber', 'pf_nopw_l-huber_op']
    for feat in feats:

        lr = 'lr_' + str(lr_n).replace('.', '-')
        print(f"lr: {lr_n}, feat: {feat}")
        
        psnrs = []
        for scene in scenes:
            scene_dir = f'{results_root}/{model}/{scene}/{lr}/{feat}/psnr.txt'
            with open(scene_dir, 'r') as f:
                data = f.readlines()
                for d in data:
                    if 'mean' in d:
                        psnrs.append(float(d.split(' ')[-1]))
        avg = sum(psnrs) / len(psnrs)
        cat_psnrs = [f'{scenes[i]}: {psnrs[i]:.5f}, 'for i in range(len(scenes))]

        cat_psnrs_str = ''
        for p in cat_psnrs:
            cat_psnrs_str+=p

        cat_psnrs_str += f' avg psnr: {avg}'
        print(f'{cat_psnrs_str}')

        # print(f'{results_root}/{model}/{scene}/{lr}/{feat}-avg psnr: {avg}')

