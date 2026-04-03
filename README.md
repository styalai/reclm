uv add torch torchvision --default-index https://download.pytorch.org/whl/cu126 --frozen
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

to re install torch after tilelang
pip install torch==2.11.0 --force-reinstall --index-url https://download.pytorch.org/whl/cu126