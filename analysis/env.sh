source /home/gnoto_venvs/torch/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export W8=$HOME/ML_project/nano_gpt/weights
export D10=$HOME/ML_project/analysis/fineweb10B
export VAL10=$D10/fineweb_val_000000.bin
cd $HOME/ML_project/analysis
