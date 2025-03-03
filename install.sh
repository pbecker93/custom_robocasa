cd custom_robosuite
pip install -e .
cd ..

cd custom_robocasa
pip install -e .
cd ..

pip install -r requirements.txt

python -m robocasa.scripts.download_kitchen_assets

python -m robosuite.scripts.setup_macros
python -m robocasa.scripts.setup_macros

pip install "git+https://github.com/facebookresearch/pytorch3d.git" # takes a while