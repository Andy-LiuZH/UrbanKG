"""

The processed files are stored in ./UrbanKG
File format after alignment, filtering and preprocessing:

    entity2id.txt: entity, entity_id
    relation2id.txt: relation, relation_id
    triplets.txt：entity_id, relation_id, entity_id
    train.txt
    valid.txt
    test.txt

"""
import random
from tqdm import tqdm

def get_entity2id_relation2id(KG, entity2id, relation2id):
    entity = []
    relations = []
    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            entity.append(temp[0])
            entity.append(temp[2])
            relations.append(temp[1])
    entity = list(set(entity))
    relations = list(set(relations))
    f.close()

    with open(entity2id,'w') as f2:
        for i in range(len(entity)):
            f2.write(entity[i] + ' ')
            f2.write(str(i))
            f2.write('\n')
        f2.close()


    with open(relation2id,'w') as f3:
        for j in range(len(relations)):
            f3.write(relations[j]+' ')
            f3.write(str(j))
            f3.write('\n')
        f3.close()

def produce_train_val_test(KG, entity2id, realtion2id, triple):

    h = []
    r = []
    t = []

    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            h.append(temp[0])
            t.append(temp[2])
            r.append(temp[1])

    entity_category_dict = {}
    relation_category_dict = {}
    with open(entity2id) as f:
        for line in f.readlines():
            temp = line.split()
            entity_category_dict.update({temp[0] : temp[1]})
    with open(realtion2id) as f1:
        for line in f1.readlines():
            temp1 = line.split()
            relation_category_dict.update({temp1[0] : temp1[1]})

    with open(triple, 'w') as f2:
        for i in tqdm(range(len(h))):
            f2.write(str(entity_category_dict[h[i]]) + ' ')
            f2.write(str(relation_category_dict[r[i]]) + ' ')
            f2.write(str(entity_category_dict[t[i]]))
            f2.write('\n')


def produce_train_val_test_temporal(KG, entity2id, realtion2id, quad):

    h = []
    r = []
    t = []
    timestamps = []

    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            h.append(temp[0])
            t.append(temp[2])
            r.append(temp[1])
            timestamps.append(temp[3])

    entity_category_dict = {}
    relation_category_dict = {}

    with open(entity2id) as f:
        for line in f.readlines():
            temp = line.split()
            entity_category_dict.update({temp[0] : temp[1]})
    with open(realtion2id) as f1:
        for line in f1.readlines():
            temp1 = line.split()
            relation_category_dict.update({temp1[0] : temp1[1]})

    with open(quad, 'w') as f2:
        for i in tqdm(range(len(h))):
            f2.write(str(entity_category_dict[h[i]]) + ' ')
            f2.write(str(relation_category_dict[r[i]]) + ' ')
            f2.write(str(entity_category_dict[t[i]]) + ' ')
            f2.write(timestamps[i])
            f2.write('\n')



def get_train_val_test(triple, train_address, valid_address, test_address):
    h_r_t = []
    with open(triple) as f:
        for line in f.readlines():
            temp = line.split()
            h_r_t.append(temp)
    random.shuffle(h_r_t)
    train = int(len(h_r_t) * 0.9)
    valid = int(len(h_r_t) * 0.05)
    test = len(h_r_t) - train - valid

    with open(train_address, 'w') as f:
        for i in range(train):
            f.write(h_r_t[i][0] + '\t')
            f.write(h_r_t[i][1] + '\t')
            f.write(h_r_t[i][2])
            f.write('\n')
    f.close()

    with open(valid_address, 'w') as f:
        for i in range(valid):
            f.write(h_r_t[train + i][0] + '\t')
            f.write(h_r_t[train + i][1] + '\t')
            f.write(h_r_t[train + i][2])
            f.write('\n')
    f.close()

    with open(test_address, 'w') as f:
        for i in range(test):
            f.write(h_r_t[train + valid + i][0] + '\t')
            f.write(h_r_t[train + valid + i][1] + '\t')
            f.write(h_r_t[train + valid + i][2])
            f.write('\n')
    f.close()

def get_train_val_test_temporal(quad, train_address, valid_address, test_address):
    h_r_t_tau = []
    with open(quad) as f:
        for line in f.readlines():
            temp = line.split()
            h_r_t_tau.append(temp)

    random.shuffle(h_r_t_tau)
    train = int(len(h_r_t_tau) * 0.9)
    valid = int(len(h_r_t_tau) * 0.05)
    test = len(h_r_t_tau) - train - valid

    with open(train_address, 'w') as f:
        for i in range(train):
            f.write(h_r_t_tau[i][0] + '\t')
            f.write(h_r_t_tau[i][1] + '\t')
            f.write(h_r_t_tau[i][2] + '\t')
            f.write(h_r_t_tau[i][3])
            f.write('\n')
    f.close()


    with open(valid_address, 'w') as f:
        for i in range(valid):
            f.write(h_r_t_tau[train + i][0] + '\t')
            f.write(h_r_t_tau[train + i][1] + '\t')
            f.write(h_r_t_tau[train + i][2] + '\t')
            f.write(h_r_t_tau[i][3])
            f.write('\n')
    f.close()

    with open(test_address, 'w') as f:
        for i in range(test):
            f.write(h_r_t_tau[train + valid + i][0] + '\t')
            f.write(h_r_t_tau[train + valid + i][1] + '\t')
            f.write(h_r_t_tau[train + valid + i][2]+ '\t')
            f.write(h_r_t_tau[train + valid + i][3])
            f.write('\n')
    f.close()

if __name__ == "__main__":
    # Add # at any time based on the case
    # UrbanKG = "./UrbanKG/Static_UrbanKG/UrbanKG_Shizuoka.txt"
    # entity2id = "./UrbanKG/Static_UrbanKG/entity2id.txt"
    # relation2id = "./UrbanKG/Static_UrbanKG/relation2id.txt"
    # triple = "./UrbanKG/Static_UrbanKG/triplets.txt"
    # train = './UrbanKG/Static_UrbanKG/train.txt'
    # valid = './UrbanKG/Static_UrbanKG/valid.txt'
    # test = './UrbanKG/Static_UrbanKG/test.txt'
    #
    # get_entity2id_relation2id(UrbanKG, entity2id, relation2id)
    # produce_train_val_test(UrbanKG, entity2id, relation2id, triple)
    # get_train_val_test(triple, train, valid, test)
    #
    # UrbanKG_temporal = "./UrbanKG/Temporal_UrbanKG/T-UrbanKG_Shizuoka.txt"
    # entity2id_temporal = "./UrbanKG/Temporal_UrbanKG/T-entity2id.txt"
    # relation2id_temporal = "./UrbanKG/Temporal_UrbanKG/T-relation2id.txt"
    # quadruple = "./UrbanKG/Temporal_UrbanKG/quadruplets.txt"
    # train_temporal = './UrbanKG/Temporal_UrbanKG/T-train.txt'
    # valid_temporal = './UrbanKG/Temporal_UrbanKG/T-valid.txt'
    # test_temporal = './UrbanKG/Temporal_UrbanKG/T-test.txt'
    #
    # get_entity2id_relation2id(UrbanKG_temporal, entity2id_temporal, relation2id_temporal)
    # produce_train_val_test_temporal(UrbanKG_temporal, entity2id_temporal, relation2id_temporal, quadruple)
    # get_train_val_test_temporal(quadruple, train_temporal, valid_temporal, test_temporal)

    UrbanKG = "./UrbanKG/Shizuoka/DCM_UrbanKG/basic.txt"
    entity2id = "./UrbanKG/Shizuoka/DCM_UrbanKG/entity2id.txt"
    relation2id = "./UrbanKG/Shizuoka/DCM_UrbanKG/relation2id.txt"
    triple = "./UrbanKG/Shizuoka/DCM_UrbanKG/triplets.txt"
    train = './UrbanKG/Shizuoka/DCM_UrbanKG/train.txt'
    valid = './UrbanKG/Shizuoka/DCM_UrbanKG/valid.txt'
    test = './UrbanKG/Shizuoka/DCM_UrbanKG/test.txt'

    get_entity2id_relation2id(UrbanKG, entity2id, relation2id)
    produce_train_val_test(UrbanKG, entity2id, relation2id, triple)
    get_train_val_test(triple, train, valid, test)