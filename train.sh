clear
export TORCH_CUDA_ARCH_LIST="8.6"
NUM_PROCESSES=3
PYTHON_SCRIPT="run.py"
SCRIPT_ARGS=""

accelerate launch \
    --num_processes ${NUM_PROCESSES} \
    ${PYTHON_SCRIPT} $@
