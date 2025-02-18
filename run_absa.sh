#!/usr/bin/env bash
# echo "--tfm_type mbert \
#             --exp_type acs \
#             --model_name_or_path $mbert_path \
#             --data_dir ./data \
#             --src_lang en \
#             --tgt_lang fr \
#             --do_train \
#             --do_eval \
#             --ignore_cached_data \
#             --per_gpu_train_batch_size 16 \
#             --per_gpu_eval_batch_size 16 \
#             --learning_rate 5e-5 \
#             --tagging_schema BIEOS \
#             --overwrite_output_dir \
#             --max_steps 2000 \
#             --train_begin_saving_step 1500 \
#             --eval_begin_end 1500-2000 
# "
# mbert_path=../trained-transformers/bert-multi-cased
mbert_path=/data/wuchengyan/XABSA-master/mBERT
# xlmr_path=../trained-transformers/xlmr-base
xlmr_path=/data/wuchengyan/XABSA-master/XLM-R

python /data/wuchengyan/XABSA-master/main.py --tfm_type xlmr \
            --exp_type supervised \
            --model_name_or_path $mbert_path \
            --data_dir ./data1 \
            --src_lang fr \
            --tgt_lang fr \
            --do_train \
            --do_eval \
            --ignore_cached_data \
            --per_gpu_train_batch_size 8 \
            --per_gpu_eval_batch_size 8 \
            --learning_rate 2e-5 \
            --tagging_schema BIEOS \
            --overwrite_output_dir \
            --max_steps 2500 \
            --train_begin_saving_step 1500 \
            --eval_begin_end 2000-2500 
