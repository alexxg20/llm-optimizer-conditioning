# Copy to env.sh and adapt the paths. Load with:  source env.sh
source ~/venvs/torch/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export W8=/home/ML_project/nano_gpt/weights                        # model_stepNNNNNN.safetensors
export D10=/home/ML_project/analysis/fineweb10B                    # FineWeb-10B shards
export VAL10=$D10/fineweb_val_000000.bin
export VAL=/home/ML_project/analysis/fineweb100B/fineweb_val_000000.bin       # early tests only
export TRAIN5=/home/ML_project/analysis/fineweb100B/fineweb_train_000005.bin  # early tests only
cd /home/ML_project/analysis
