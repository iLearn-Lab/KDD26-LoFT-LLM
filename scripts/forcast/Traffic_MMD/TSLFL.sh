export CUDA_VISIBLE_DEVICES=0

# model_name=DLinear_TSLFL
model_name=iTransformer_TSLFL
plens=(8)
seeds=(1)

for seed in ${seeds[@]}; do
  for pl in ${plens[@]}; do
    python -u run_plfm.py \
      --task_name preModel \
      --is_training 1 \
      --root_path ./dataset/Traffic_TimeMMD/ \
      --data_path Traffic_TimeMMD.csv \
      --model_id Traffic_TimeMMD_8_$pl \
      --model PLFM \
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
      --out_patch_len 4 \
      --seed $seed \
      --patience 10 \
      --train_epochs 20 \
      --learning_rate 0.001

    python -u run_TSLFL.py \
      --task_name TSLFL_Forcast \
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
      --out_patch_len 4 \
      --TSLFL_patch_len 4 \
      --seed $seed \
      --patience 10 \
      --train_epochs 20 \
      --learning_rate 0.0002
  done
done