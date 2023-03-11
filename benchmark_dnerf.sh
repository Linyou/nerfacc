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

# for scene in $scenes; do 
# if [ "$scene" = "$exception" ]; then
#     lr=0.003
#     echo running $scene : --use_feat_predict, lr: $lr
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene --use_feat_predict --lr $lr
# else
#     lr=0.01
#     echo running $scene : --use_feat_predict, lr: $lr
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene --use_feat_predict  --lr $lr
# fi
# done

# python mean_psnr.py

# for scene in $scenes; do 
#     echo ---------- running $scene ----------
#     lr=0.01
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -ms 1024 --test_print 
# done


# benchmark for diff learning rate
# for scene in $scenes; do 
#     echo ---------- running $scene ----------
#     lr=0.02
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -ms 1024 --test_print 
# done

# for scene in $scenes; do 
#     echo ---------- running $scene ----------
#     lr=0.015
#     python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -ms 1024 --test_print 
# done
END=2
for hl in $(seq 2 $END); do

    # for scene in $scenes; do 
    #     echo ---------- running $scene ----------
    #     lr=0.01
    #     python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -te -ta -ms 64 -hl $hl
    # done

    # for scene in $scenes; do 
    #     echo ---------- running $scene ----------
    #     lr=0.01
    #     python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -te -ta -ms 128 -hl $hl
    # done

    for scene in $scenes; do 
        echo ---------- running $scene ----------
        lr=0.01
        python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -te -ta -ms 256 -hl $hl
    done

    for scene in $scenes; do 
        echo ---------- running $scene ----------
        lr=0.01
        python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -te -ta -ms 512 -hl $hl
    done

    for scene in $scenes; do 
        echo ---------- running $scene ----------
        lr=0.01
        python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -te -ta -ms 1024 -hl $hl
    done

    for scene in $scenes; do 
        echo ---------- running $scene ----------
        lr=0.01
        python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -te -ta -ms 2048 -hl $hl
    done

    for scene in $scenes; do 
        echo ---------- running $scene ----------
        lr=0.01
        python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -te -ta -ms 4096 -hl $hl
    done

    for scene in $scenes; do 
        echo ---------- running $scene ----------
        lr=0.01
        python examples/train_ngp_dnerf.py --train_split train --scene $scene --lr $lr --rec_loss huber -f -o -df -te -ta -ms 8192 -hl $hl
    done

done