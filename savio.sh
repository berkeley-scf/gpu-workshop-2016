pledge # use 1x2
ssh paciorek@hpc.brc.berkeley.edu


module unload intel
module load cuda

sacctmgr -p show associations user=paciorek

srun -A ac_scsguest -p savio2_gpu  -N 1 -t 30:0 --pty bash
srun -u -A ac_scsguest -p savio2_gpu  -N 1 -t 30:0 bash -i

alias gtop=\"nvidia-smi -q -d UTILIZATION -l 1\"
alias gmem=\"nvidia-smi -q -d MEMORY -l 1\"

nvcc ${CUDA_DIR}/samples/1_Utilities/deviceQuery/deviceQuery.cpp -I${CUDA_DIR}/include -I${CUDA_DIR}/samples/common/inc -o deviceQuery

