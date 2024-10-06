CUDA_VISIBLE_DEVICES=0 \
    accelerate launch \
    --mixed_precision=bf16 \
    --num_processed=1 \
    --main_process_port=$(expr $RANDOM + 1000) \
    run_rerank.py