#!/usr/bin/bash
printf "\n\nRunning: temperature distribution plots combined:\n"
python ./temperature_distributions/plot_temperature_distribution.py -s TNG300-1 --load-data --combine --show-divisions
printf "\n\nRunning: temperature distribution plots grid:\n"
python ./temperature_distributions/plot_temperature_distribution.py -s TNG300-1 --load-data --grid --show-divisions
printf "\n\nRunning: temperature distribution plots grid (mass weighted):\n"
python ./temperature_distributions/plot_temperature_distribution.py -s TNG300-1 --load-data --grid --show-divisions --use-mass
printf "\n\nRunning: temperature distribution plots grid (temperature normalized):\n"
python ./temperature_distributions/plot_temperature_distribution.py -s TNG300-1 --load-data --grid --show-divisions --normalize-temperatures

printf "\n\nRunning: mass trends plots (median):\n"
python ./mass_trends/plot_mass_trends.py -s TNG300-1 --load-data
printf "\n\nRunning: mass trends plots (mean):\n"
python ./mass_trends/plot_mass_trends.py -s TNG300-1 --load-data --use-average
printf "\n\nRunning: mass trends plots (median, curve):\n"
python ./mass_trends/plot_mass_trends.py -s TNG300-1 --load-data --running-median
printf "\n\nRunning: mass trends plots (mean, curve):\n"
python ./mass_trends/plot_mass_trends.py -s TNG300-1 --load-data --use-average --running-median

printf "\n\nRunning: radial profile plots (TNG300, temperature, mean):\n"
python ./radial_profiles/stack_individual_radial_profiles.py -s TNG300-1 --what temperature --method mean --log
printf "\n\nRunning: radial profile plots (TNG300, temperature, median):\n"
python ./radial_profiles/stack_individual_radial_profiles.py -s TNG300-1 --what temperature --method median --log
printf "\n\nRunning: radial profile plots (TNG300, density, mean):\n"
python ./radial_profiles/stack_individual_radial_profiles.py -s TNG300-1 --what density --method mean --log
printf "\n\nRunning: radial profile plots (TNG300, density, median):\n"
python ./radial_profiles/stack_individual_radial_profiles.py -s TNG300-1 --what density --method median --log
printf "\n\nRunning: radial profile plots (TNG-Cluster, temperature, mean):\n"
python ./radial_profiles/stack_individual_radial_profiles.py -s TNG-Cluster --what temperature --method mean --log
printf "\n\nRunning: radial profile plots (TNG-Cluster, temperature, median):\n"
python ./radial_profiles/stack_individual_radial_profiles.py -s TNG-Cluster --what temperature --method median --log
printf "\n\nRunning: radial profile plots (TNG-Cluster, density, mean):\n"
python ./radial_profiles/stack_individual_radial_profiles.py -s TNG-Cluster --what density --method mean --log
printf "\n\nRunning: radial profile plots (TNG-Cluster, density, median):\n"
python ./radial_profiles/stack_individual_radial_profiles.py -s TNG-Cluster --what density --method median --log

printf "\n\nRunning: radial profile plots (all clusters, temperature, mean):\n"
python ./radial_profiles/stackbin_radial_profiles.py --what temperature --method mean --log
printf "\n\nRunning: radial profile plots (all clusters, temperature, median):\n"
python ./radial_profiles/stackbin_radial_profiles.py --what temperature --method median --log
printf "\n\nRunning: radial profile plots (all clusters, density, mean):\n"
python ./radial_profiles/stackbin_radial_profiles.py --what density --method mean --log
printf "\n\nRunning: radial profile plots (all clusters, density, median):\n"
python ./radial_profiles/stackbin_radial_profiles.py --what density --method median --log