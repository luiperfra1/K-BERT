import os
import subprocess

for i in range(1, 11):
    log_filename = f"./outputs/salud_mental/conKG/salud_mental{i}_conKG.log"
    print(f"Ejecutando experimento {i}, guardando en {log_filename}...")

    command = f"""
    CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_cls_ingles.py \
        --pretrained_model_path ./models/google_model_en_uncased_base.bin \
        --config_path ./models/google_config.json \
        --vocab_path ./models/google_uncased_en_vocab.txt \
        --train_path ./datasets/salud_mental/train.tsv \
        --dev_path ./datasets/salud_mental/dev.tsv \
        --test_path ./datasets/salud_mental/test.tsv \
        --epochs_num 5 --batch_size 32 \
        --kg_name brain/kgs/conceptnet_salud_mental.spo \
        --seed none \
        --output_model_path ./outputs/salud_mental_conKG.bin \
        > {log_filename} 2>&1
    """
    subprocess.call(command, shell=True)
    print(f"✅ Terminado el experimento {i}\n")