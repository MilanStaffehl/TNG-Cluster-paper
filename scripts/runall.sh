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

printf "\n\nRunning: cluster mass trend plots (full cluster, SFR):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field SFR -r
printf "\n\nRunning: cluster mass trend plots (full cluster, total BH mass):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field TotalBHMass -r
printf "\n\nRunning: cluster mass trend plots (full cluster, BH mass):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHMass -r
printf "\n\nRunning: cluster mass trend plots (full cluster, BH accretion rate):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHMdot -r
printf "\n\nRunning: cluster mass trend plots (full cluster, BH mode):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHMode -r
printf "\n\nRunning: cluster mass trend plots (full cluster, gas metallicity):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --field GasMetallicity -r
printf "\n\nRunning: cluster mass trend plots (full cluster, BH cumulative energy):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHCumEnergy -r
printf "\n\nRunning: cluster mass trend plots (full cluster, BH cumulative mass):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHCumMass -r

printf "\n\nRunning: cluster mass trend plots (cluster core only, SFR):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field SFR -r -cc
printf "\n\nRunning: cluster mass trend plots (cluster core only, total BH mass):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field TotalBHMass -r -cc
printf "\n\nRunning: cluster mass trend plots (cluster core only, BH mass):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHMass -r -cc
printf "\n\nRunning: cluster mass trend plots (cluster core only, BH accretion rate):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHMdot -r -cc
printf "\n\nRunning: cluster mass trend plots (cluster core only, BH mode):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHMode -r -cc
printf "\n\nRunning: cluster mass trend plots (cluster core only, gas metallicity):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --field GasMetallicity -r -cc
printf "\n\nRunning: cluster mass trend plots (cluster core only, BH cumulative energy):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHCumEnergy -r -cc
printf "\n\nRunning: cluster mass trend plots (cluster core only, BH cumulative mass):\n"
python ./mass_trends/plot_cool_gas_mass_trends.py --log --color-log --field BHCumMass -r -cc