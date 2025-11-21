for SEED in 100;
do python train_semeval.py --model_name_or_path bert-base-cased  --output_dir ./models/semeval --input_format entity_marker --num_train_epochs 10.0 --seed $SEED --train_batch_size 32 --test_batch_size 32 --learning_rate 2e-5 --gradient_accumulation_steps 1 --run_name bert-base;
done;