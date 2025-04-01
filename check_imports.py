import os
import warnings

warnings.filterwarnings("ignore")

# os.system("conda")

# os.system("conda info --envs")
# env = input(
#     "If you can see Conda environments\nType the environment name you want to choose.\nIf you cant see any names or dont want to choose any environement and want to continue on your main python environment just enter by letting it stay empty or choose a random wrong name.\nName: "
# )
# print(env)
# print(len(env))
# if len(env) > 0:
#     print("continue with " + env)
#     os.system("conda activate" + env)
# else:
#     print("continue without environment")


try:
    import pandas
except ImportError:
    print("installing pandas")
    os.system("pip install pandas")

try:
    import aiohttp
except ImportError:
    print("installing aiohttp")
    os.system("pip install aiohttp")

try:
    import asyncio
except ImportError:
    print("installing asyncio")
    os.system("pip install asyncio")

try:
    import datetime
except ImportError:
    print("installing datetime")
    os.system("pip install datetime")

try:
    import numpy
except ImportError:
    print("installing numpy")
    os.system("pip install numpy")

try:
    import progress
except ImportError:
    print("installing progress")
    os.system("pip install progress")

try:
    import tqdm
except ImportError:
    print("installing tqdm")
    os.system("pip install tqdm")

try:
    import torch
except ImportError:
    print("installing torch")
    print(
        "if torch for cpu isnt enough please install it using 'https://pytorch.org/get-started/locally/'"
    )
    os.system("pip install torch torchvision torchaudio")

try:
    import matplotlib
except ImportError:
    print("installing matplotlib")
    os.system("pip install matplotlib")

try:
    import PIL
except ImportError:
    print("installing pillow")
    os.system("pip install pillow")

try:
    import gdown
except ImportError:
    print("installing gdown")
    os.system("pip install gdown")

try:
    import flet
except ImportError:
    print("installing flet")
    os.system("pip install flet")
