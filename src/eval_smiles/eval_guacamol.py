from guacamol.assess_distribution_learning import assess_distribution_learning
import logging

# 创建 logger
logger = logging.getLogger(__name__)  # 主模块 logger
logger.setLevel(logging.INFO)  # 设置日志级别

# 创建 handler，输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置日志格式
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)

# 添加 handler 到 logger
logger.addHandler(console_handler)

library_logger = logging.getLogger('guacamol')  # 用库的模块名
library_logger.handlers = logger.handlers  # 共享 main 的 handlers
library_logger.setLevel(logger.level)  # 共享 main 的日志级别

smiles_list = [s.strip() for s in open('/data2/chensm22/HRS/MHdiff/sample/denovo/guacamol/charged1/final_smiles.txt').readlines()]
results = assess_distribution_learning(smiles_list,
                                       chembl_training_file='/data2/chensm22/HRS/data/guacamol/guacamol_pyg/raw/guacamol_v1_train.smiles')