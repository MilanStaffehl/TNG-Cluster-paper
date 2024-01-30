#!/usrcd/bin/bash
python ./temperature_distributions/plot_temperature_distribution.py -s MAIN_SIM --load-data --combine --show-divisions
python ./temperature_distributions/plot_temperature_distribution.py -s MAIN_SIM --load-data --grid --show-divisions
python ./temperature_distributions/plot_temperature_distribution.py -s MAIN_SIM --load-data --grid --show-divisions --use-mass_trends
python ./temperature_distributions/plot_temperature_distribution.py -s MAIN_SIM --load-data --grid --show-divisions --normalize-temperatures

python ./mass_trends/plot_mass_trends.py -s MAIN_SIM --load-data

python ./radial_profiles/stack_individual_radial_profiles.py -s MAIN_SIM --what temperature --method mean --log
python ./radial_profiles/stack_individual_radial_profiles.py -s MAIN_SIM --what temperature --method median --log
python ./radial_profiles/stack_individual_radial_profiles.py -s MAIN_SIM --what density --method mean --log
python ./radial_profiles/stack_individual_radial_profiles.py -s MAIN_SIM --what density --method median --log
python ./radial_profiles/stack_individual_radial_profiles.py -s CLUSTER --what temperature --method mean --log
python ./radial_profiles/stack_individual_radial_profiles.py -s CLUSTER --what temperature --method median --log
python ./radial_profiles/stack_individual_radial_profiles.py -s CLUSTER --what density --method mean --log
python ./radial_profiles/stack_individual_radial_profiles.py -s CLUSTER --what density --method median --log

python ./radial_profiles/stackbin_radial_profiles.py --what temperature --method mean --log
python ./radial_profiles/stackbin_radial_profiles.py --what temperature --method median --log
python ./radial_profiles/stackbin_radial_profiles.py --what density --method mean --log
python ./radial_profiles/stackbin_radial_profiles.py --what density --method median --log