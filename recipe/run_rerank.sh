CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --num_processed=4 \
    --main_process_port=$(expr $RANDOM + 1000) \
    run_rerank.py