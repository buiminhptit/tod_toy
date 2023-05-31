
#!/bin/bash#
python3 curio.py --data_path "data/merged_aq.json" --base_model "PygmalionAI/pygmalion-6b" \
  --finetune_method 'qlora' --lora_r 8 --lora_alpha 32 --output_dir 'out/PygmalionCurio' \
    --batch_size=128 --micro_batch_size 4 --cutoff_len 512 --num_epochs 3 --load_in_8bit False