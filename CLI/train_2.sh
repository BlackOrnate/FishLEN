export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 ../train.py \
  --image_path "../Dataset/AC I/train/Images_padded_output" \
  --mask_path "../Dataset/AC I/train/Masks_padded_output" \
  --csv_path "../Dataset/AC I/aging_cohort_I.csv" \
  --save_path "./results/2" \
  --extra_info_column_names "sex" "day_index"\
  --gpu 4