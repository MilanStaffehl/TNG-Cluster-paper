from pathlib import Path
import logging
import logging.config
import sys
import time

# import the helper scripts
cur_dir = Path(__file__).parent.resolve()
sys.path.append(str(cur_dir.parent / "src"))
import plotters
import logging_config


logging_cfg = logging_config.get_logging_config("INFO")
logging.config.dictConfig(logging_cfg)
logger = logging.getLogger("root")
# test setup
logger.info("I am a test log!")

# sim data
TEST_SIM = "TNG50-3"
DEV_SIM = "TNG300-2"
MAIN_SIM = "TNG300-1"

# plot hist data
MASS_BINS = [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
hist_plotter = plotters.TemperatureDistributionPlotter(
    TEST_SIM, MASS_BINS, logger
)
begin = time.time()
hist_plotter.get_data()
hist_plotter.get_mask()
hist_plotter.get_hists_lin()
end = time.time()
logger.info(f"Spent {end - begin:.2f} seconds on execution.")

for i in range(len(MASS_BINS) - 1):
    hist_plotter.plot_stacked_hist(i)
