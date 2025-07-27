from argparse import ArgumentParser
from pathlib import Path
from si_gui_phy.core import analyzer_from_phy
from spikeinterface_gui import run_mainwindow

parser = ArgumentParser()
parser.add_argument('phy_path')
phy_path = Path(parser.parse_args().phy_path)

if phy_path.is_dir() is False:
    raise FileExistsError(f"Path {phy_path} is not a folder.")

sa = analyzer_from_phy(phy_path=phy_path)

run_mainwindow(sa, mode="desktop", layout_preset="legacy")
