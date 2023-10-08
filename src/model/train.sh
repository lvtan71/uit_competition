python model/train.py --outdir checkpoint/retrieval_model \
--train_path data/pairs_data_train \
--valid_path data/pairs_data_valid \
--bert_pretrain vinai/phobert-base-v2 \
--num_train_epochs 10 \
--eval_step 200 \
--max_len 256 \
--gradient_accumulation_steps 1