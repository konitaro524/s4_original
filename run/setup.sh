#!/usr/bin/env bash
set -e

# 1) Python の仮想環境（venv）を作る
python3 -m venv .venv
source .venv/bin/activate

# 2) requirements.txt を使って pip install
pip install --upgrade pip
pip install -r requirements.txt

# 3) 必要なら npm / yarn も同様に
# npm install
# yarn install

# 4) あとはいつものスクリプトを実行
# python train.py など
