#!/bin/bash
# Pre-registered transfer: DP-SGD recipes chosen on Fashion-MNIST (rounds 1-3) applied unchanged to MNIST (and
# re-seeded on Fashion), plus the Gaussian-prior PAC-Bayes pipeline on MNIST for comparison with published results.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
P=~/.venv/bin/python; S=experiments/dp_sgd_bound.py
# Gaussian-prior PAC-Bayes on MNIST (block estimator, delta = 0.025): ours (bn_stat + inv + klinv01) and the
# Perez-Ortiz-style baseline (plain parameterization, f_quad), width 64, sigma0 = 0.05, 3 seeds
$P experiments/pacbayes_norm.py --data /tmp/lv_mnist_std.npz --out results/v2/mnist_pb_ours.jsonl --norms bn_stat,none \
   --modes inv --widths 64 --s0s 0.05 --objective klinv01 --seeds 0,1,2 --workers 6 --threads 4 > logs/mnist_pb_ours.log 2>&1 &
$P experiments/pacbayes_norm.py --data /tmp/lv_mnist_std.npz --out results/v2/mnist_pb_quad.jsonl --norms none \
   --modes plain --widths 64 --s0s 0.05 --objective quad --seeds 0,1,2 --workers 3 --threads 4 > logs/mnist_pb_quad.log 2>&1 &
for D in mnist fashion; do
  [ $D = mnist ] && SEEDS=0,1,2 || SEEDS=1,2
  # recipe A: linear softmax on Gabor features, nu 0.01, B 4096, 10 epochs, lr 0.3, cosine
  $P $S --data /tmp/lv_${D}_raw.npz --features gabor --out results/v2/dp_transfer_${D}_A.jsonl --depths 0 --norms ln \
     --batches 4096 --nus 0.01 --epochs 10 --lrs 0.3 --seeds $SEEDS --workers 3 --threads 4 > logs/dp_transfer_${D}_A.log 2>&1 &
  # recipe B: 1-hidden-layer LN MLP (128) on Gabor features, nu 0.01, B 4096, 5 epochs, lr 1, cosine
  $P $S --data /tmp/lv_${D}_raw.npz --features gabor --out results/v2/dp_transfer_${D}_B.jsonl --depths 1 --widths 128 \
     --norms ln --batches 4096 --nus 0.01 --epochs 5 --lrs 1 --seeds $SEEDS --workers 3 --threads 4 > logs/dp_transfer_${D}_B.log 2>&1 &
done
wait
touch logs/transfer_done
