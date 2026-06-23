aggle---competition
项目简介：
本项目为Kaggle二分类竞赛，基于F1赛事数据预测赛车下一圈是否进站。使用LightGBM、XGBoost、CatBoost三模型加权融合，完成数据清洗与时序特征工程。
测试集AUC：0.944
竞赛排名：1108/1942

文件结构
.gitignore
README.md
kaggle.py  # 完整实验代码
submission_3model_final.csv  # 最终提交文件

运行方式
1. 下载竞赛数据集放到项目目录
2. 安装依赖：
pip install numpy pandas lightgbm xgboost catboost scikit-learn
3. 执行代码：python kaggle.py

仓库地址
GitHub：https://github.com/202481313711/kaggle---competition

