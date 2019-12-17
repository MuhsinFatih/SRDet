CONTAINER=container/sandboxdir
sing="singularity exec --nv $CONTAINER "
(\
export CUDA_VISIBLE_DEVICES=0
JOBNAME=input48_interleaved
$sing python3 -u 'train.py' --inputsize 48 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
) &
(\
export CUDA_VISIBLE_DEVICES=1
JOBNAME=input24_interleaved
$sing python3 -u 'train.py' --inputsize 24 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
) &

wait