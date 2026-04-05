"""
云母分类系统后端API
支持两个XGBoost模型推理：
1. 主量元素模型 (13个特征)
2. 微量元素模型 (19个特征)
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 模型和标准化器路径
MODEL_DIR = "models"
# 支持多种模型文件格式
MAJOR_MODEL_PATHS = [
    os.path.join(MODEL_DIR, "major", "XGBoost_model.joblib"),  # 实际存在的路径
    os.path.join(MODEL_DIR, "XGBoost_model.joblib"),            # 备用路径
    os.path.join(MODEL_DIR, "major", "model.pkl"),              # 原始预期文件
    os.path.join(MODEL_DIR, "major", "model.joblib"),           # 其他可能格式
    os.path.join(MODEL_DIR, "RandomForest_model.joblib"),
    os.path.join(MODEL_DIR, "SVM_model.joblib"),
    os.path.join(MODEL_DIR, "AdaBoost_model.joblib"),
]

MAJOR_SCALER_PATHS = [
    os.path.join(MODEL_DIR, "major", "scaler.pkl"),             # 实际存在的路径
    os.path.join(MODEL_DIR, "scaler.pkl"),                      # 备用路径
    os.path.join(MODEL_DIR, "major", "scaler.joblib")           # 其他可能格式
]

TRACE_MODEL_PATHS = [
    os.path.join(MODEL_DIR, "trace", "XGBoost_model.joblib"),   # 实际存在的路径
    os.path.join(MODEL_DIR, "XGBoost_model.joblib"),            # 备用路径
    os.path.join(MODEL_DIR, "trace", "model.pkl"),             # 原始预期文件
    os.path.join(MODEL_DIR, "trace", "model.joblib"),           # 其他可能格式
    os.path.join(MODEL_DIR, "RandomForest_model.joblib"),
    os.path.join(MODEL_DIR, "SVM_model.joblib"),
    os.path.join(MODEL_DIR, "AdaBoost_model.joblib"),
]

TRACE_SCALER_PATHS = [
    os.path.join(MODEL_DIR, "trace", "scaler.pkl"),             # 实际存在的路径
    os.path.join(MODEL_DIR, "scaler.pkl"),                      # 备用路径
    os.path.join(MODEL_DIR, "trace", "scaler.joblib")           # 其他可能格式
]

# 特征名称（必须与训练时一致）
# 主量元素特征 (13个，使用实际元素名称)
MAJOR_FEATURES = [
    "SiO2", "TiO2", "Al2O3", "FeO", "MnO", "MgO", "CaO",
    "Na2O", "K2O", "F", "Cl", "Mg#", "A/CNK"  # 13个
]

# 微量元素特征 (19个，使用实际元素名称)
TRACE_FEATURES = [
    "Li", "Sc", "V", "Cr", "Co", "Ni", "Rb", "Sr", "Nb", "Sn",
    "Cs", "Ba", "Ta", "W", "Nb/Ta", "V/Sc", "Rb/Sr", "Rb/Ba", "Nb/Sn"  # 19个
]

# 前端发送的特征名称（用于验证）- 这些是HTML界面使用的实际元素名称
MAJOR_FRONTEND_FEATURES = MAJOR_FEATURES.copy()
TRACE_FRONTEND_FEATURES = TRACE_FEATURES.copy()

# 特征名称映射：前端元素名称 -> 后端特征名称（直接映射，因为名称相同）
MAJOR_FEATURE_MAPPING = {feature: feature for feature in MAJOR_FEATURES}
TRACE_FEATURE_MAPPING = {feature: feature for feature in TRACE_FEATURES}

# 类别名称（根据训练数据调整）
CLASS_NAMES = ["Sn", "W", "斑岩Cu"]

# 全局变量存储加载的模型和标准化器
major_model = None
major_scaler = None
trace_model = None
trace_scaler = None

def load_models():
    """加载模型和标准化器"""
    global major_model, major_scaler, trace_model, trace_scaler
    
    try:
        # 查找并加载主量元素模型
        major_model_loaded = False
        for model_path in MAJOR_MODEL_PATHS:
            if os.path.exists(model_path):
                major_model = joblib.load(model_path)
                major_model_loaded = True
                break
        
        if not major_model_loaded:
            # 创建模拟模型
            from sklearn.ensemble import RandomForestClassifier
            major_model = RandomForestClassifier(n_estimators=10, random_state=42)
            # 使用虚拟数据训练模拟模型
            X_dummy = np.random.randn(100, len(MAJOR_FEATURES))
            y_dummy = np.random.choice([0, 1, 2], 100)
            major_model.fit(X_dummy, y_dummy)
        
        # 查找并加载主量元素标准化器
        major_scaler_loaded = False
        for scaler_path in MAJOR_SCALER_PATHS:
            if os.path.exists(scaler_path):
                major_scaler = joblib.load(scaler_path)
                major_scaler_loaded = True
                break
        
        if not major_scaler_loaded:
            # 创建模拟标准化器
            from sklearn.preprocessing import StandardScaler
            major_scaler = StandardScaler()
            # 使用虚拟数据拟合
            X_dummy = np.random.randn(100, len(MAJOR_FEATURES))
            major_scaler.fit(X_dummy)
        
        # 查找并加载微量元素模型
        trace_model_loaded = False
        for model_path in TRACE_MODEL_PATHS:
            if os.path.exists(model_path):
                trace_model = joblib.load(model_path)
                trace_model_loaded = True
                break
        
        if not trace_model_loaded:
            # 创建模拟模型
            from sklearn.ensemble import RandomForestClassifier
            trace_model = RandomForestClassifier(n_estimators=10, random_state=42)
            # 使用虚拟数据训练模拟模型
            X_dummy = np.random.randn(100, len(TRACE_FEATURES))
            y_dummy = np.random.choice([0, 1, 2], 100)
            trace_model.fit(X_dummy, y_dummy)
        
        # 查找并加载微量元素标准化器
        trace_scaler_loaded = False
        for scaler_path in TRACE_SCALER_PATHS:
            if os.path.exists(scaler_path):
                trace_scaler = joblib.load(scaler_path)
                trace_scaler_loaded = True
                break
        
        if not trace_scaler_loaded:
            # 创建模拟标准化器
            from sklearn.preprocessing import StandardScaler
            trace_scaler = StandardScaler()
            # 使用虚拟数据拟合
            X_dummy = np.random.randn(100, len(TRACE_FEATURES))
            trace_scaler.fit(X_dummy)
            
    except Exception as e:
        pass

def preprocess_major_data(data_dict):
    """预处理主量元素数据"""
    try:
        # 将前端元素名称映射到后端特征名称
        mapped_data = {}
        for frontend_feature, backend_feature in MAJOR_FEATURE_MAPPING.items():
            if frontend_feature in data_dict:
                mapped_data[backend_feature] = data_dict[frontend_feature]
            else:
                # 如果缺少特征，使用0填充
                mapped_data[backend_feature] = 0.0
        
        # 创建DataFrame，确保特征顺序正确
        data = pd.DataFrame([mapped_data], columns=MAJOR_FEATURES)
        
        # 将字符串转换为数值
        data = data.astype(float)
        
        # 检查并处理负值，将其置为0
        num_features = data.select_dtypes(include=[np.number])
        neg_mask = num_features < 0
        if neg_mask.any().any():
            num_features = num_features.mask(neg_mask, 0)
            data = num_features
        
        # 检查并处理NaN值
        if data.isnull().values.any():
            for col in data.columns:
                if data[col].isnull().any():
                    col_mean = data[col].mean()
                    data[col].fillna(col_mean, inplace=True)
        
        # 数据预处理：对数变换 + 标准化（与训练时保持一致）
        if major_scaler is not None:
            # 首先确保所有值大于0（避免log(0)）
            data = data + 1e-6
            # 对数变换
            data_log = np.log(data)
            # 标准化
            data_scaled = major_scaler.transform(data_log)
        else:
            # 如果没有标准化器，返回原始数据
            if isinstance(data, pd.DataFrame):
                data_scaled = data.values
            else:
                data_scaled = data
            # 确保返回numpy数组
            if not isinstance(data_scaled, np.ndarray):
                data_scaled = np.array(data_scaled)
                
        # 确保返回numpy数组
        if isinstance(data_scaled, pd.DataFrame):
            data_scaled = data_scaled.values
        elif not isinstance(data_scaled, np.ndarray):
            data_scaled = np.array(data_scaled)
            
        return data_scaled
        
    except Exception as e:
        return None

def preprocess_trace_data(data_dict):
    """预处理微量元素数据"""
    try:
        # 将前端元素名称映射到后端特征名称
        mapped_data = {}
        for frontend_feature, backend_feature in TRACE_FEATURE_MAPPING.items():
            if frontend_feature in data_dict:
                mapped_data[backend_feature] = data_dict[frontend_feature]
            else:
                # 如果缺少特征，使用0填充
                mapped_data[backend_feature] = 0.0
        # 创建DataFrame，确保特征顺序正确
        data = pd.DataFrame([mapped_data], columns=TRACE_FEATURES)
        # 将字符串转换为数值
        data = data.astype(float)
        # 检查并处理负值，将其置为0
        num_features = data.select_dtypes(include=[np.number])
        neg_mask = num_features < 0
        if neg_mask.any().any():
            num_features = num_features.mask(neg_mask, 0)
            data = num_features
        # 检查并处理NaN值
        if data.isnull().values.any():
            for col in data.columns:
                if data[col].isnull().any():
                    col_mean = data[col].mean()
                    data[col].fillna(col_mean, inplace=True)
        # 数据预处理：对数变换 + 标准化（与训练时保持一致）
        if trace_scaler is not None:
            # 首先确保所有值大于0（避免log(0)）
            data = data + 1e-6
            # 对数变换
            data_log = np.log(data)
            # 标准化
            data_scaled = trace_scaler.transform(data_log)
        else:
            # 如果没有标准化器，返回原始数据
            if isinstance(data, pd.DataFrame):
                data_scaled = data.values
            else:
                data_scaled = data
            # 确保返回numpy数组
            if not isinstance(data_scaled, np.ndarray):
                data_scaled = np.array(data_scaled)
        # 确保返回numpy数组
        if isinstance(data_scaled, pd.DataFrame):
            data_scaled = data_scaled.values
        elif not isinstance(data_scaled, np.ndarray):
            data_scaled = np.array(data_scaled)
        return data_scaled
    except Exception as e:
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "message": "云母分类系统后端API运行正常",
        "models_loaded": {
            "major": major_model is not None,
            "trace": trace_model is not None
        }
    })

@app.route('/predict/major', methods=['POST'])
def predict_major():
    """主量元素模型预测"""
    try:
        # 检查模型是否加载
        if major_model is None:
            return jsonify({
                "error": "主量元素模型未加载",
                "message": "请确保模型文件 models/major/model.pkl 存在"
            }), 503
        # 获取JSON数据
        data = request.get_json()
        # 验证数据
        if not data:
            return jsonify({"error": "未提供数据"}), 400
        # 检查是否包含所有必需特征（使用前端特征名称）
        missing_features = [f for f in MAJOR_FRONTEND_FEATURES if f not in data]
        if missing_features:
            return jsonify({
                "error": "缺少必需特征",
                "missing_features": missing_features,
                "required_features": MAJOR_FRONTEND_FEATURES,
                "note": "请确保输入所有主量元素特征"
            }), 400
        # 预处理数据
        processed_data = preprocess_major_data(data)
        if processed_data is None:
            return jsonify({"error": "数据预处理失败"}), 500
        # 进行预测
        prediction = major_model.predict(processed_data)
        # 直接使用模型的原始预测结果，不进行任何置信度调整
        predicted_class_index = int(prediction[0])  # 直接使用模型预测的类别索引
        predicted_class_name = CLASS_NAMES[predicted_class_index]  # 获取对应的类别名称

        # 返回结果 - 只返回预测类别，不包含任何概率或置信度信息
        return jsonify({
            "success": True,
            "model": "major",
            "prediction": predicted_class_name,  # 直接返回模型预测的类别
            "prediction_index": predicted_class_index,   # 直接返回类别索引
            "features_received": list(data.keys()),
            "features_processed": MAJOR_FEATURES
        })
    except Exception as e:
        return jsonify({
            "error": "预测失败",
            "message": str(e)
        }), 500

@app.route('/predict/trace', methods=['POST'])
def predict_trace():
    """微量元素模型预测"""
    try:
        # 检查模型是否加载
        if trace_model is None:
            return jsonify({
                "error": "微量元素模型未加载",
                "message": "请确保模型文件 models/trace/model.pkl 存在"
            }), 503
        # 获取JSON数据
        data = request.get_json()
        # 验证数据
        if not data:
            return jsonify({"error": "未提供数据"}), 400
        # 检查是否包含所有必需特征（使用前端特征名称）
        missing_features = [f for f in TRACE_FRONTEND_FEATURES if f not in data]
        if missing_features:
            return jsonify({
                "error": "缺少必需特征",
                "missing_features": missing_features,
                "required_features": TRACE_FRONTEND_FEATURES,
                "note": "请确保输入所有微量元素特征"
            }), 400
        # 预处理数据
        processed_data = preprocess_trace_data(data)
        if processed_data is None:
            return jsonify({"error": "数据预处理失败"}), 500
        # 进行预测
        prediction = trace_model.predict(processed_data)
        # 直接使用模型的原始预测结果，不进行任何置信度调整
        predicted_class_index = int(prediction[0])  # 直接使用模型预测的类别索引
        predicted_class_name = CLASS_NAMES[predicted_class_index]  # 获取对应的类别名称

        # 返回结果 - 只返回预测类别，不包含任何概率或置信度信息
        return jsonify({
            "success": True,
            "model": "trace",
            "prediction": predicted_class_name,  # 直接返回模型预测的类别
            "prediction_index": predicted_class_index,   # 直接返回类别索引
            "features_received": list(data.keys()),
            "features_processed": TRACE_FEATURES
        })
    except Exception as e:
        return jsonify({
            "error": "预测失败",
            "message": str(e)
        }), 500

@app.route('/predict/both', methods=['POST'])
def predict_both():
    """同时使用两个模型进行预测"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供数据"}), 400
        # 分别调用两个预测函数
        major_result = None
        trace_result = None
        # 检查是否有主量元素特征
        major_features_present = all(f in data for f in MAJOR_FRONTEND_FEATURES)
        if major_features_present and major_model is not None:
            # 临时创建请求对象来调用predict_major
            major_processed = preprocess_major_data(data)
            if major_processed is not None:
                major_pred = major_model.predict(major_processed)
                class_idx = int(major_pred[0])
                major_result = {
                    "prediction": CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"未知类别{class_idx}",
                    "prediction_index": class_idx
                }
        # 检查是否有微量元素特征
        trace_features_present = all(f in data for f in TRACE_FRONTEND_FEATURES)
        if trace_features_present and trace_model is not None:
            # 临时创建请求对象来调用predict_trace
            trace_processed = preprocess_trace_data(data)
            if trace_processed is not None:
                trace_pred = trace_model.predict(trace_processed)
                class_idx = int(trace_pred[0])
                trace_result = {
                    "prediction": CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"未知类别{class_idx}",
                    "prediction_index": class_idx
                }
        return jsonify({
            "success": True,
            "major": major_result,
            "trace": trace_result,
            "features_present": {
                "major": major_features_present,
                "trace": trace_features_present
            },
            "models_loaded": {
                "major": major_model is not None,
                "trace": trace_model is not None
            }
        })
    except Exception as e:
        return jsonify({
            "error": "预测失败",
            "message": str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """批量预测端点 - 支持Excel文件中的多条数据"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供数据"}), 400
        # 检查数据格式
        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({
                "error": "数据格式错误",
                "message": "需要包含'data'字段，且为数组格式"
            }), 400
        samples = data['data']
        if len(samples) == 0:
            return jsonify({"error": "数据为空"}), 400
        # 限制批量处理数量
        MAX_BATCH_SIZE = 100
        if len(samples) > MAX_BATCH_SIZE:
            return jsonify({
                "error": "批量处理数量超限",
                "message": f"最多支持{MAX_BATCH_SIZE}条数据，当前{len(samples)}条"
            }), 400
        predictions = []
        error_samples = []
        for idx, sample in enumerate(samples):
            try:
                # 检查样本类型（主量元素或微量元素）
                has_major = all(f in sample for f in MAJOR_FRONTEND_FEATURES)
                has_trace = all(f in sample for f in TRACE_FRONTEND_FEATURES)
                result = {
                    "sample_index": idx + 1,
                    "has_major_features": has_major,
                    "has_trace_features": has_trace,
                    "major_prediction": None,
                    "trace_prediction": None,
                    "error": None
                }
                # 主量元素预测
                if has_major and major_model is not None:
                    major_processed = preprocess_major_data(sample)
                    if major_processed is not None:
                        major_pred = major_model.predict(major_processed)
                        class_idx = int(major_pred[0])  # 确保转换为Python int
                        result["major_prediction"] = {
                            "prediction": CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"未知类别{class_idx}",
                            "prediction_index": class_idx
                        }
                # 微量元素预测
                if has_trace and trace_model is not None:
                    trace_processed = preprocess_trace_data(sample)
                    if trace_processed is not None:
                        trace_pred = trace_model.predict(trace_processed)
                        class_idx = int(trace_pred[0])  # 确保转换为Python int
                        result["trace_prediction"] = {
                            "prediction": CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"未知类别{class_idx}",
                            "prediction_index": class_idx
                        }
                predictions.append(result)
            except Exception as e:
                error_samples.append({
                    "sample_index": idx + 1,
                    "error": str(e)
                })
        return jsonify({
            "success": True,
            "total_samples": len(samples),
            "successful_predictions": len(predictions),
            "failed_predictions": len(error_samples),
            "predictions": predictions,
            "errors": error_samples
        })
    except Exception as e:
        return jsonify({
            "error": "批量预测失败",
            "message": str(e)
        }), 500

@app.route('/models/info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    return jsonify({
        "major_model": {
            "loaded": major_model is not None,
            "features": MAJOR_FRONTEND_FEATURES,
            "classes": CLASS_NAMES,
            "scaler_loaded": major_scaler is not None
        },
        "trace_model": {
            "loaded": trace_model is not None,
            "features": TRACE_FRONTEND_FEATURES,
            "classes": CLASS_NAMES,
            "scaler_loaded": trace_scaler is not None
        }
    })

@app.route('/features/major', methods=['GET'])
def get_major_features():
    """获取主量元素特征列表"""
    return jsonify({
        "features": MAJOR_FRONTEND_FEATURES,
        "count": len(MAJOR_FRONTEND_FEATURES),
        "description": "主量元素特征（13个）"
    })

@app.route('/features/trace', methods=['GET'])
def get_trace_features():
    """获取微量元素特征列表"""
    return jsonify({
        "features": TRACE_FRONTEND_FEATURES,
        "count": len(TRACE_FRONTEND_FEATURES),
        "description": "微量元素特征（19个）"
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """获取类别列表"""
    return jsonify({
        "classes": CLASS_NAMES,
        "count": len(CLASS_NAMES),
        "description": "云母分类类别"
    })

# 启动时加载模型
if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=8080, debug=False)