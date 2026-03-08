export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3

outdir="/fast/yangz16/outputs/x-win/vitbase_nlstdrr_mimic"
logdir="$outdir/log"

if [[ ! -d "$outdir" ]]; then
  mkdir -p "$outdir"
  echo "Created directory: $outdir"
else
  echo "Directory already exists: $outdir"
fi

torchrun --nproc_per_node=4 train.py \
--warmup 10 \
--epochs 100 \
--bs 24 \
--train_txt /fast/yangz16/outputs/x-win/train_drrs3.txt \
--test_txt /fast/yangz16/outputs/x-win/test_drrs3.txt \
--train_real_txt /fast/yangz16/outputs/x-win/train_mimic.txt \
--test_real_txt /fast/yangz16/outputs/x-win/test_mimic.txt \
--loss_type contrastive \
--contrastive_temp 0.1 \
--recon_weight 1.0 \
--outdir "$outdir" \
&> "$logdir" &
