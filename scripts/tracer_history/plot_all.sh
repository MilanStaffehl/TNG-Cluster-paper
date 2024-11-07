#!/bin/bash
# STACKS PHYSICAL UNITS
printf "\n\nRunning: distance plots (stacks, all plot types)"
python ./plot_quantity_with_time.py distance -l -q
printf "\n\nRunning: volume normalized distance plots (stacks, all plot types)"
python ./plot_quantity_with_time.py distance -l -vn -q
printf "\n\nRunning: temperature plots (stacks, all plot types)"
python ./plot_quantity_with_time.py temperature -l -q
printf "\n\nRunning: density plots (stacks, all plot types)"
python ./plot_quantity_with_time.py density -l -q

# STACKS NORMALIZED
printf "\n\nRunning: distance plots (normalized stacks, all plot types)"
python ./plot_quantity_with_time.py distance -l -n -q
printf "\n\nRunning: volume normalized distance plots (normalized stacks, all plot types)"
python ./plot_quantity_with_time.py distance -l -vn -n -q
printf "\n\nRunning: temperature plots (normalized stacks, all plot types)"
python ./plot_quantity_with_time.py temperature -l -n -q

# STACKS SPLIT BY PARENT
printf "\n\nRunning: distance plots (stacks, split by parent, all plot types)"
python ./plot_quantity_with_time.py distance -l -pt 1 --split-by parent-category -q
printf "\n\nRunning: volume normalized distance plots (stacks, split by parent, all plot types)"
python ./plot_quantity_with_time.py distance -l -vn -pt 1 --split-by parent-category -q
printf "\n\nRunning: temperature plots (stacks, split by parent, all plot types)"
python ./plot_quantity_with_time.py temperature -l -pt 1 --split-by parent-category -q
printf "\n\nRunning: density plots (stacks, split by parent, all plot types)"
python ./plot_quantity_with_time.py density -l -pt 1 --split-by parent-category -q

# STACKS SPLIT BY PARENT AT Z = 0
printf "\n\nRunning: distance plots (stacks, split by parent, all plot types)"
python ./plot_quantity_with_time.py distance -l -pt 1 --split-by parent-category-at-zero -q
printf "\n\nRunning: volume normalized distance plots (stacks, split by parent, all plot types)"
python ./plot_quantity_with_time.py distance -l -vn -pt 1 --split-by parent-category-at-zero -q
printf "\n\nRunning: temperature plots (stacks, split by parent, all plot types)"
python ./plot_quantity_with_time.py temperature -l -pt 1 --split-by parent-category-at-zero -q
printf "\n\nRunning: density plots (stacks, split by parent, all plot types)"
python ./plot_quantity_with_time.py density -l -pt 1 --split-by parent-category-at-zero -q

# STACKS SPLIT BY BOUND STATE
printf "\n\nRunning: distance plots (stacks, split by bound state, all plot types)"
python ./plot_quantity_with_time.py distance -l -pt 1 --split-by bound-state -q
printf "\n\nRunning: volume normalized distance plots (stacks, split by bound state, all plot types)"
python ./plot_quantity_with_time.py distance -l -vn -pt 1 --split-by bound-state -q
printf "\n\nRunning: temperature plots (stacks, split by bound state, all plot types)"
python ./plot_quantity_with_time.py temperature -l -pt 1 --split-by bound-state -q
printf "\n\nRunning: density plots (stacks, split by bound state, all plot types)"
python ./plot_quantity_with_time.py density -l -pt 1 --split-by bound-state -q

# STACKS SPLIT BY BOUND STATE AT Z = 0
printf "\n\nRunning: distance plots (stacks, split by bound state, all plot types)"
python ./plot_quantity_with_time.py distance -l -pt 1 --split-by bound-state-at-zero -q
printf "\n\nRunning: volume normalized distance plots (stacks, split by bound state, all plot types)"
python ./plot_quantity_with_time.py distance -l -vn -pt 1 --split-by bound-state-at-zero -q
printf "\n\nRunning: temperature plots (stacks, split by bound state, all plot types)"
python ./plot_quantity_with_time.py temperature -l -pt 1 --split-by bound-state-at-zero -q
printf "\n\nRunning: density plots (stacks, split by bound state, all plot types)"
python ./plot_quantity_with_time.py density -l -pt 1 --split-by bound-state-at-zero -q

# INDIVIDUAL PLOTS
for i in $(seq 0 351)
do
  printf "\n\nRunning: distance plots (individual, zoom-in $i)\n"
  python ./plot_quantity_with_time.py distance -l -z "$i" -q
  printf "\n\nRunning: volume normalized distance plots (individual, zoom-in $i)\n"
  python ./plot_quantity_with_time.py distance -l -z "$i" -vn -q
  printf "\n\nRunning: temperature plots (individual, zoom-in $i)\n"
  python ./plot_quantity_with_time.py temperature -l -z "$i" -q
  printf "\n\nRunning: density plots (individual, zoom-in $i)\n"
  python ./plot_quantity_with_time.py density -l -z "$i" -q
done

# INDIVIDUAL LINE PLOTS, SPLIT BY CATEGORY AT REDSHIFT ZERO
for i in $(seq 0 351)
do
  printf "\n\nRunning: distance line plots (individual, zoom-in $i, split by parent category)\n"
  python ./plot_quantity_with_time.py distance -l -z "$i" -pt 0 --split-by parent-category-at-zero -q
  printf "\n\nRunning: temperature line plots (individual, zoom-in $i, split by parent category)\n"
  python ./plot_quantity_with_time.py temperature -l -z "$i" -pt 0 --split-by parent-category-at-zero -q
  printf "\n\nRunning: density line plots (individual, zoom-in $i, split by parent category)\n"
  python ./plot_quantity_with_time.py density -l -z "$i" -pt 0 --split-by parent-category-at-zero -q
done

# CROSSING TIME PLOTS
printf "\n\nRunning: crossing time plots (all plot types)"
python ./plot_crossing_time_plots.py -q

# PARENT CATEGORY PLOTS
printf "\n\nRunning: parent category plots (all plot types)"
python ./plot_parent_category_plots.py -q

printf "\n\nDone! Finished all plots."