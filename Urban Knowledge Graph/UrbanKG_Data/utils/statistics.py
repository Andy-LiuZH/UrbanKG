
Ent = []
Rel = []
Triplet = []
Train = []
Valid = []
Test = []

with open("../UrbanKG/Shizuoka/DCM_UrbanKG/entity2id.txt") as f:
    for line in f.readlines():
        Ent.append(line)

print("Ent:", len(Ent))

with open("../UrbanKG/NYC/relation2id_NYC.txt") as f:
    for line in f.readlines():
        Rel.append(line)
print("Rel:", len(Rel))

with open("../UrbanKG/NYC/triplets_NYC.txt") as f:
    for line in f.readlines():
        Triplet.append(line)
print("Triplet:", len(Triplet))

with open("../UrbanKG/NYC/train_NYC.txt") as f:
    for line in f.readlines():
        Train.append(line)
print("Train:", len(Train))

with open("../UrbanKG/NYC/valid_NYC.txt") as f:
    for line in f.readlines():
        Valid.append(line)
print("Valid:", len(Valid))

with open("../UrbanKG/NYC/test_NYC.txt") as f:
    for line in f.readlines():
        Test.append(line)
print("Test:", len(Test))



