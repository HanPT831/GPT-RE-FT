for SEED in 0;
do python train_scierc.py --model_name_or_path 'allenai/scibert_scivocab_uncased'  --output_dir ./models/scierc --input_format entity_marker --num_train_epochs 10.0 --seed $SEED --train_batch_size 32 --test_batch_size 32 --learning_rate 2e-5 --gradient_accumulation_steps 1 --run_name bert-scierc;
done;