1. 
`data_preprocessing_Shizuoka.py`
处理原数据

2. 
`processing_functions.py`
处理数据用的函数
在`1`的函数里调用

3. 
`construct_ST-UrbanKG_Shizuoka.py`
对于处理好的数据，以<实体1，关系，实体2>的三元组形式储存为UrbanKG.txt
对于OD数据，以<实体1，关系，实体2，时间戳>的四元组形式储存为Temporal UrbanKG.txt

4. 
`data_split_en2id_rel2id.py`
对于静态/时态图谱，分割数据集，且创建实体及关系索引
	`get_entity2id_relation2id`：
	给实体和关系赋予ID
	`produce_train_val_test(temporal)`：
	从KG和实体、关系中分割出训练验证测试集
	`get_train_val_test(temporal)`：
	获取上述创建的训练验证测试集

运行1->3->4