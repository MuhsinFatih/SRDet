CONTAINER=container/sandboxdir
sing="singularity exec --nv $CONTAINER "
(\
export CUDA_VISIBLE_DEVICES=0
JOBNAME=input96
$sing python3 -u 'train.py' --inputsize 96 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
) # &
(\
export CUDA_VISIBLE_DEVICES=0
JOBNAME=input64
$sing python3 -u 'train.py' --inputsize 64 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
) # &
(\
export CUDA_VISIBLE_DEVICES=0
JOBNAME=input48
$sing python3 -u 'train.py' --inputsize 48 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
) # &
(\
export CUDA_VISIBLE_DEVICES=0
JOBNAME=input36
$sing python3 -u 'train.py' --inputsize 36 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
) # &