printf "\n\nPearson CC (lin)\n"
python plot_statistical_measures.py -w pcc
printf "\n\nPearson CC (log)\n"
python plot_statistical_measures.py -w pcc-log
printf "\n\nPearson CC (log-log)\n"
python plot_statistical_measures.py -w pcc-loglog
printf "\n\nRatio\n"
python plot_statistical_measures.py -w ratio

printf "\n\nPearson CC (lin) - cluster core\n"
python plot_statistical_measures.py -w pcc -cc
printf "\n\nPearson CC (log) - cluster core\n"
python plot_statistical_measures.py -w pcc-log -cc
printf "\n\nPearson CC (log-log) - cluster core\n"
python plot_statistical_measures.py -w pcc-loglog -cc
printf "\n\nRatio - cluster core\n"
python plot_statistical_measures.py -w ratio -cc