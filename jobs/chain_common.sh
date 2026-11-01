cd ~/asrbench && source venv/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=8
s(){ echo "### $1 done $(date +%H:%M)"; }
