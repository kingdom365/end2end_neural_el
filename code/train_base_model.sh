#!/bin/bash

# 创建一个函数来执行训练任务
execute_training() {
    python3 -m model.train \
        --batch_size=4 \
        --experiment_name=corefmerge \
        --training_name=group_lstm/base_model_v"$1" \
        --ent_vecs_regularization=l2dropout \
        --evaluation_minutes=10 \
        --nepoch_no_imprv=6 \
        --span_emb="boundaries" \
        --dim_char=50 \
        --hidden_size_char=50 \
        --hidden_size_lstm=150 \
        --nn_components=pem_lstm \
        --fast_evaluation=True \
        --all_spans_training=True \
        --final_score_ffnn=0_0 \
        --train_datasets=aida_train \
        --el_datasets=aida_dev_z_aida_test_z_aida_train \
        --el_val_datasets=0
}

# 为不同的模型版本执行训练任务
for v in 1 2 3
do
    execute_training "$v"
done
