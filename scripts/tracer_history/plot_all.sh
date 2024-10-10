#!/bin/bash
# STACKS PHYSICAL UNITS
printf "Running: distance plots (stacks, all plot types)"
python ./plot_quantity_with_time.py distance -l
printf "Running: volume normalized distance plots (stacks, all plot types)"
python ./plot_quantity_with_time.py distance -l -vn
printf "Running: temperature plots (stacks, all plot types)"
python ./plot_quantity_with_time.py temperature -l
printf "Running: density plots (stacks, all plot types)"
python ./plot_quantity_with_time.py density -l

# STACKS NORMALIZED
printf "Running: distance plots (normalized stacks, all plot types)"
python ./plot_quantity_with_time.py distance -l -n
printf "Running: volume normalized distance plots (normalized stacks, all plot types)"
python ./plot_quantity_with_time.py distance -l -vn -n
printf "Running: temperature plots (normalized stacks, all plot types)"
python ./plot_quantity_with_time.py temperature -l -n

# INDIVIDUAL PLOTS
for i in $(seq 0 351)
do
  printf "\n\nRunning: distance plots (individual, zoom-in $i)\n"
  python ./plot_quantity_with_time.py distance -l -z "$i"
  printf "\n\nRunning: volume normalized distance plots (individual, zoom-in $i)\n"
  python ./plot_quantity_with_time.py distance -l -z "$i" -vn
  printf "\n\nRunning: temperature plots (individual, zoom-in $i)\n"
  python ./plot_quantity_with_time.py temperature -l -z "$i"
  printf "\n\nRunning: density plots (individual, zoom-in $i)\n"
  python ./plot_quantity_with_time.py density -l -z "$i"
done