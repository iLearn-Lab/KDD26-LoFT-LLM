export CUDA_VISIBLE_DEVICES=4

# model_name=iTransformer
# model_name=FITS
# model_name=TimesNet
# model_name=PatchTST
model_name=DLinear
# model_name=FreTS
# model_name=TimeXer
# model_name=Transformer

plens=(8)
# plens=(1 2 3 4 5 6 7 8 9 10)
seeds=(1)


for seed in ${seeds[@]}; do
  for pl in ${plens[@]}; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/Traffic_TimeMMD/ \
      --data_path Traffic_TimeMMD.csv \
      --model_id Traffic_TimeMMD_8_$pl \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 8 \
      --label_len 0 \
      --pred_len $pl \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1 \
      --d_model 256 \
      --d_ff 256 \
      --seed $seed
  done
done


  # python -u run.py \
  # --task_name long_term_forecast \
  # --is_training 1 \
  # --root_path ./dataset/ETT-small/ \
  # --data_path ETTh1.csv \
  # --model_id ETTh1_96_96 \
  # --model $model_name \
  # --data ETTh1 \
  # --features M \
  # --seq_len 96 \
  # --label_len 0 \
  # --pred_len 96 \
  # --e_layers 2 \
  # --d_layers 1 \
  # --factor 3 \
  # --enc_in 7 \
  # --dec_in 7 \
  # --c_out 7 \
  # --des 'Exp' \
  # --itr 1

