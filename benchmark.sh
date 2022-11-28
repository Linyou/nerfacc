# python examples/train_ngp_dnerf.py --train_split train --scene lego 


scenes="bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex"
exception="bouncingballs"
lr=0.01

#version 1

# feat: --use_feat_predict, --use_feat_predict_mix

# for scene in $scenes; do 
# if [ "$scene" = "$exception" ]; then
#     lr=0.003
#     echo running $scene without predict feature with lr: $lr
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr
#     echo running $scene with predict feature  with lr: $lr
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene --use_feat_predict --lr $lr
# else
#     lr=0.01
#     echo running $scene without predict feature with lr: $lr
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene  --lr $lr
#     echo running $scene with predict feature with lr: $lr
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene --use_feat_predict  --lr $lr
# fi
# done

#version 2

# feat: --use_feat_predict, --use_feat_predict_mix

for scene in $scenes; do 
if [ "$scene" = "$exception" ]; then
    lr=0.003
    echo running $scene : --use_feat_predict, lr: $lr
    python examples/train_ngp_dnerf.py --train_split train --scene $scene --use_feat_predict --lr $lr
else
    lr=0.01
    echo running $scene : --use_feat_predict, lr: $lr
    python examples/train_ngp_dnerf.py --train_split train --scene $scene --use_feat_predict  --lr $lr
fi
done

python mean_psnr.py