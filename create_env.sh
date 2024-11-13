# 检查是否存在名为imner的conda环境
env_exists=$(conda env list | grep 'imner')

if [ -z "$env_exists" ]; then
    # 如果环境不存在，创建名为imner的conda环境，并指定Python版本为3.10
    echo "Creating a conda environment named 'imner' with Python 3.10..."
    conda create --name imner python=3.10 -y
    conda activate imner
    pip install -r requirements.txt
    sudo apt-get update
    #安装soundfile
    sudo apt-get install libsndfile1
    #安装sox相关
    sudo apt-get install sox
    sudo apt install libsox-dev
else
    # 如果环境已存在，输出环境已存在的消息
    echo "环境'imner'已经存在！"
    conda activate imner
fi





