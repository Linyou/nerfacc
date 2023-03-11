# python examples/train_ngp_dnerf.py --train_split train --scene lego 


scenes="flame_salmon_1 coffee_martini cook_spinach cut_roasted_beef flame_steak sear_steak"
for scene in $scenes; do 
    echo ---------- running $scene ----------
    python examples/train_dngp_nerf_multires_f16.py --scene $scene --data_factor 2 -ms 0.0001 --max_steps 40000 -gn 6 -ds 3dnerf -df -te -f --lr 0.005
done