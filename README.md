# si-gui-phy

This package aims to support opening Kilosort4/Phy output using spikeinterface-gui.

To use this package, first clone the github repo and change into its directory:

```
git clone https://github.com/chrishalcrow/si-gui-phy.git
cd si-gui-phy
```

## Using uv 

If you use `uv` (https://docs.astral.sh/uv/getting-started/installation/), the following should just work:

```
uv run python -i sigui_from_phy.py path/to/phy_folder
```

## Using pip/venv

If not, please set up a new virtual environment, then add this package. If you're in the directory you've cloned from
github, run

```
pip install -e .
```

Now, from the same directory, run

```
python sigui_from_phy.py path/to/phy/output
```