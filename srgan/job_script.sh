CONTAINER=container/sandboxdir
sing="singularity exec --nv $CONTAINER "
# (\
# export CUDA_VISIBLE_DEVICES=1
# JOBNAME=input96
# $sing python3 -u 'train.py' --inputsize 96 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
# ) &
(\
export CUDA_VISIBLE_DEVICES=1
JOBNAME=input48
$sing python3 -u 'train.py' --inputsize 48 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
) &
# (\
# export CUDA_VISIBLE_DEVICES=2
# JOBNAME=input24
# $sing python3 -u 'train.py' --inputsize 24 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
# ) &
# (\
# export CUDA_VISIBLE_DEVICES=3
# JOBNAME=input12
# $sing python3 -u 'train.py' --inputsize 12 --exp $JOBNAME >> outputs/$JOBNAME.log 2>&1
# ) &

wait