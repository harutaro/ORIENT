# ORIENT

##############################################
#  This GIT directory is under construction  #
##############################################

## Directories
RESULT_AAAI2025/	Logs
script/			scripts for experiment
src/			source code (modified version of FACIL)
venv/			Virtual Environmet including sow_cpp package

## Setup
(1) install torch environment (already done in this GIT)
$ cd $(TOP)
$ python3 -m venv venv
$ source venv/bin/activate.csh
$ python3 -c 'import sys; print("\n".join(sys.path))'
$ pip install -U pip
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
$ pip install matplotlib scipy pynvml
$ (not mandatory)
  $ pip install torchinfo
  $ pip install openpyxl pynvml
(2) check 'sow_cpp' package (already done in this GIT)
$ cd $(TOP)
$ cat /mnt/public/ORIENT/aaai2025/venv/lib/python3.9/site-packages/sow_cpp-0.9.5-py3.9-linux-x86_64.egg/EGG-INFO/PKG-INFO
(3) setup initial parameters of ORIENT (already done in this GIT)
$ cd $(TOP)
$ source venv/bin/activate.csh
$ cd src
$ mkdir networks/INIT
$ python save_tbl.py ${gpu} 10
$ python save_S.py ${gpu} 10
  -- on_the_fly --
$ python save_V.py ${gpu} 10 512 20 0.90 1.0
$ python save_U.py ${gpu} 10 512 20 0.90 1.0
(4) setup Log Directory
$ cd $(TOP)
$ mkdir RESULT_AAAI2025/

## Execution of experiments
(1)
