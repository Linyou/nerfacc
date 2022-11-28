results_root = '/home/loyot/workspace/code/training_results/nerfacc'
model = 'ngp_dnerf'
scenes=["bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex"]

feats = ['no_pf', 'pf', 'pf_mix']
lrs = ['lr_0-003', 'lr_0-01']

scenes_lr = {
    "bouncingballs":1e-2, 
    "hellwarrior":1e-2, 
    "hook": 1e-2, 
    "jumpingjacks": 1e-2, 
    "lego": 1e-2, 
    "mutant": 1e-2, 
    "standup": 5e-3, 
    "trex": 5e-3}
for lr in lrs[-1:]:
    for feat in feats:
        psnrs = []
        for scene in scenes:
            if scene == 'bouncingballs':
                scene_dir = f'{results_root}/{model}/{scene}/{lrs[0]}/{feat}/psnr.txt'
            else:
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
        print(f'{cat_psnrs_str}')

        print(f'{results_root}/{model}/{scene}/{lr}/{feat}-avg psnr: {avg}')

