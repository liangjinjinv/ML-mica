@echo off
chcp 65001 >nul
echo 正在安装依赖包...

REM 首先尝试安装wheel以支持二进制包
pip install wheel

REM 逐个安装依赖包，避免numpy编译问题
echo 安装基础依赖...
pip install setuptools wheel

echo 安装numpy...
pip install numpy==1.24.3

echo 安装pandas...
pip install pandas==2.0.3

echo 安装scikit-learn...
pip install scikit-learn==1.3.0

echo 安装其他依赖...
pip install Flask==2.3.3 Flask-CORS==4.0.0 joblib==1.3.2 scipy==1.11.2

echo 依赖包安装完成！
pause