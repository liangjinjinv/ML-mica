@echo off
chcp 65001 >nul
title 云母分类系统 - 一键启动
color 0A

echo.
echo ========================================
echo       云母分类系统 - 一键启动
echo ========================================
echo.

REM 检查Python是否安装
echo [信息] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.7+
    echo.
    echo 解决方案：
    echo 1. 安装Python 3.7或更高版本
    echo 2. 或使用独立运行包（无需安装Python）
    echo.
    pause
    exit /b 1
)

REM 检查依赖包
echo [信息] 检查依赖包...
pip list | findstr "flask" >nul
if errorlevel 1 (
    echo [信息] 安装依赖包...
    echo [提示] 如果安装失败，请运行 install_dependencies.bat 手动安装
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
) else (
    echo [信息] 依赖包已安装
)

REM 检查模型文件是否存在
echo [信息] 检查模型文件...
if not exist "models\major\XGBoost_model.joblib" (
    echo [警告] 主量元素模型文件不存在: models\major\XGBoost_model.joblib
    echo [信息] 系统将使用模拟模型进行演示
) else (
    echo [信息] 主量元素模型文件存在
)

if not exist "models\major\scaler.pkl" (
    echo [警告] 主量元素标准化器文件不存在: models\major\scaler.pkl
    echo [信息] 系统将使用模拟标准化器（预测结果可能不准确）
) else (
    echo [信息] 主量元素标准化器文件存在
)

if not exist "models\trace\XGBoost_model.joblib" (
    echo [警告] 微量元素模型文件不存在: models\trace\XGBoost_model.joblib
    echo [信息] 系统将使用模拟模型进行演示
) else (
    echo [信息] 微量元素模型文件存在
)

if not exist "models\trace\scaler.pkl" (
    echo [警告] 微量元素标准化器文件不存在: models\trace\scaler.pkl
    echo [信息] 系统将使用模拟标准化器（预测结果可能不准确）
) else (
    echo [信息] 微量元素标准化器文件存在
)

REM 创建必要的目录（如果不存在）
if not exist "models\major" mkdir "models\major" >nul 2>&1
if not exist "models\trace" mkdir "models\trace" >nul 2>&1

REM 启动后端服务器
echo.
echo [信息] 启动后端服务器...
echo [信息] 请勿关闭此窗口
echo.
echo [信息] 服务器启动后，请打开浏览器访问：
echo [信息] http://localhost:8080
echo.
echo [信息] 或直接双击 "云母分类.html" 文件打开Web界面
echo.
echo [信息] 按 Ctrl+C 可停止服务器
echo.
echo ========================================
echo.

REM 在新的窗口中打开前端界面
echo [信息] 打开Web界面...
start "" "云母分类.html" >nul 2>&1

REM 启动后端服务器
echo [信息] 启动后端API服务（端口8080）...
python backend.py

echo [信息] 服务器已停止
pause