from pathlib import Path
from si_gui_phy.core import analyzer_from_phy
phy_path = Path("/Users/christopherhalcrow/Work/Harry_Project/derivatives/M25/D20/sorter_output")
sa = analyzer_from_phy(phy_path=phy_path)

from spikeinterface_gui import run_mainwindow
run_mainwindow(sa, mode="desktop", layout_preset="legacy")