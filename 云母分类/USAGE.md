# 云母分类系统 - 使用版

## 系统说明

这是一个精简版的云母分类系统，专为实际使用而优化，删除了所有调试信息和日志打印，只保留核心功能。

## 文件结构

```
云母分类系统_使用版/
├── backend.py          # 精简版后端API
├── 云母分类.html       # 原版前端界面（已删除判别图解可视化部分）
├── 一键启动.bat       # 启动脚本
├── install_dependencies.bat  # 依赖安装脚本
├── requirements.txt    # 依赖包列表
├── README.md          # 说明文档
├── models/            # 模型文件目录
│   ├── major/         # 主量元素模型
│   │   ├── XGBoost_model.joblib
│   │   └── scaler.pkl
│   └── trace/         # 微量元素模型
│       ├── XGBoost_model.joblib
│       └── scaler.pkl
└── USAGE.md           # 本使用说明
```

## 功能特性

### 预测模式
1. **单条输入**：分别进行主量元素和微量元素预测
2. **Excel批量导入**：支持Excel/CSV文件批量处理

### 输入特征
- **主量元素**（13个）：SiO2, TiO2, Al2O3, FeO, MnO, MgO, CaO, Na2O, K2O, F, Cl, Mg#, A/CNK
- **微量元素**（19个）：Li, Sc, V, Cr, Co, Ni, Rb, Sr, Nb, Sn, Cs, Ba, Ta, W, Nb/Ta, V/Sc, Rb/Sr, Rb/Ba, Nb/Sn

### 输出类别
- Sn（锡型）
- W（钨型）  
- 斑岩Cu（斑岩铜型）

## 使用方法

### 方法一：一键启动（推荐）
1. 双击运行 `一键启动.bat` 文件
2. 系统会自动检查并安装依赖包
3. 自动启动后端服务并打开前端界面
4. 在界面中输入相应的元素含量数据进行预测

### 方法二：手动安装依赖
如果一键启动失败，请按以下步骤操作：

1. 运行 `install_dependencies.bat` 手动安装依赖包
2. 确认所有依赖包安装成功后
3. 打开命令行，进入此目录
4. 运行 `python backend.py` 启动后端服务
5. 打开 `云母分类.html` 使用前端界面

### 方法三：命令行启动
```
cd "云母分类系统_使用版"
python backend.py
```
然后在浏览器中访问 `http://localhost:8080`

## 依赖包

运行前请确保安装以下依赖：
```
Flask==2.3.3
Flask-CORS==4.0.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
scipy==1.11.2
```

## API端点

- `GET /health` - 健康检查
- `POST /predict/major` - 主量元素预测
- `POST /predict/trace` - 微量元素预测
- `POST /predict/batch` - 批量预测

## 注意事项

- 确保模型文件存在，否则系统会使用模拟模型
- 输入数据时请确保所有必需特征都有值
- 预测结果直接显示模型原始输出，无额外处理
- 如果遇到numpy安装问题，请使用 `install_dependencies.bat` 脚本
- 系统需要Python 3.7+ 环境

## 故障排除

### 依赖安装失败
- 运行 `install_dependencies.bat` 脚本
- 或手动逐个安装：`pip install numpy pandas scikit-learn flask`

### 后端无法启动
- 检查端口8080是否被占用
- 确认Python环境和依赖包安装正确

### 前端无法连接后端
- 确认后端服务正在运行
- 检查网络连接和防火墙设置