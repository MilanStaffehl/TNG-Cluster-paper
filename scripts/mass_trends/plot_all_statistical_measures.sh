printf "\n\nPearson CC (gas fraction, lin)\n"
python plot_statistical_measures.py -w pcc
printf "\n\nPearson CC (gas fraction, log)\n"
python plot_statistical_measures.py -w pcc-log
printf "\n\nPearson CC (gas fraction, log-log)\n"
python plot_statistical_measures.py -w pcc-loglog
printf "\n\nRatio (gas fraction)\n"
python plot_statistical_measures.py -w ratio

printf "\n\nPearson CC (gas fraction, lin) - cluster core\n"
python plot_statistical_measures.py -w pcc -cc
printf "\n\nPearson CC (gas fraction, log) - cluster core\n"
python plot_statistical_measures.py -w pcc-log -cc
printf "\n\nPearson CC (gas fraction, log-log) - cluster core\n"
python plot_statistical_measures.py -w pcc-loglog -cc
printf "\n\nRatio - cluster core (gas fraction)\n"
python plot_statistical_measures.py -w ratio -cc

printf "\n\nPearson CC (gas mass, lin)\n"
python plot_statistical_measures.py -w pcc --absolute-mass
printf "\n\nPearson CC (gas mass, log)\n"
python plot_statistical_measures.py -w pcc-log --absolute-mass
printf "\n\nPearson CC (gas mass, log-log)\n"
python plot_statistical_measures.py -w pcc-loglog --absolute-mass
printf "\n\nRatio (gas mass)\n"
python plot_statistical_measures.py -w ratio --absolute-mass

printf "\n\nPearson CC (gas mass, lin) - cluster core\n"
python plot_statistical_measures.py -w pcc -cc --absolute-mass
printf "\n\nPearson CC (gas mass, log) - cluster core\n"
python plot_statistical_measures.py -w pcc-log -cc --absolute-mass
printf "\n\nPearson CC (gas mass, log-log) - cluster core\n"
python plot_statistical_measures.py -w pcc-loglog -cc --absolute-mass
printf "\n\nRatio - cluster core (gas mass)\n"
python plot_statistical_measures.py -w ratio -cc --absolute-mass