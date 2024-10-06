CUDA_VISIBLE_DEVICES=0 \
    accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --num_processed=1 \
    --main_process_port=$(expr $RANDOM + 1000) \
    run_retrieve.py