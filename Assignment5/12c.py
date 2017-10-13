import math

child_node_1 = 8
child_node_2 = 1
child_node_1_true = 4
child_node_2_true = 0
initial_entropy = 0.991

child_node_1_false = child_node_1 - child_node_1_true
child_node_2_false = child_node_2 - child_node_2_true
ratio_1 = child_node_1_true/float(child_node_1)
ratio_2 = child_node_2_true/float(child_node_2)

if (ratio_1 == 0 or ratio_1 == 1):
    ent_child_1 = 0
else:
    ent_child_1 = -(ratio_1) * math.log(ratio_1, 2) - \
                  (1 - ratio_1) * math.log(1 - ratio_1, 2)
if (ratio_2 == 0 or ratio_2 == 1):
    ent_child_2 = 0
else:
    ent_child_2 = -(ratio_2) * math.log(ratio_2, 2) - \
                  (1 - ratio_2) * math.log(1 - ratio_2, 2)

print(ent_child_1)
print(ent_child_2)

weighted_entropy = ent_child_1*child_node_1/float(9) + ent_child_2*child_node_2/float(9)

info_gain = initial_entropy - weighted_entropy
print (info_gain)
