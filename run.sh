

# CHANGEME - path to your environment python (when environment is activated write "which python")
python_exec= < python_path >

JOB_SCRIPT="PYTHONPATH=. $python_exec ../main.py \
        --model_type="roberta-large"  \
        --data_dir="data_folder_path"  \
        --train_file_name="train_file_name"  \
        --test_file_name="test_file_name"  \
        --val_file_name="val_file_name"  \
        --batch_size=4 \
        --train_epoch_num=1 \
        --learning_rate=3e-4 \
        --max_input_num=128   \
        --max_grad_norm=2  \
        --do_shuffle \
        --loss_ratio=0.5 \
        --task_num=4  \
"

