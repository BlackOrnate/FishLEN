export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 ../test.py \
  --image_path "../Dataset/AC I/full/Images_padded_output" \
  --csv_path "../Dataset/AC I/aging_cohort_I.csv" \
  --save_path "./results/1" \
  --gpu 3