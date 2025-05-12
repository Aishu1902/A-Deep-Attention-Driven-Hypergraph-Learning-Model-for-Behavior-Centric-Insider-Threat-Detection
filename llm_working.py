from torch_geometric.data import Data
import torch
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import trange
from hybrid_new import ResHybNet
from early_stop_v1 import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
f1_L = ['helly']
f2_L = ['llm_seq74']
feat_list = [f1 + '+' + f2 for f1 in f1_L for f2 in f2_L]
LR =0.0001
train_round = 5
EPOCH =50
#isolated_points_list is actually helly points(just named it like that)
isolated_points_list = [1, 2, 3, 16, 22, 30, 46, 47, 48, 49, 51, 53, 54, 55, 59, 60, 61, 62, 65, 66, 86, 87, 88, 89, 119, 131, 132, 133, 140, 141, 142, 148, 149, 150, 151, 152, 153, 160, 163, 164, 165, 172, 198, 199, 200, 201, 202, 203, 204, 205, 211, 226, 232, 245, 246, 247, 269, 270, 271, 294, 295, 296, 304, 305, 306, 312, 313, 314, 321, 322, 323, 324, 326, 327, 328, 329, 330, 331, 334, 335, 337, 338, 339, 341, 342, 343, 344, 347, 348, 349, 350, 351, 352, 355, 356, 359, 360, 361, 362, 363, 364, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 384, 385, 386, 387, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 500, 501, 502, 503, 504, 505, 506, 507, 509, 510, 511, 512, 513, 514, 515, 516, 517, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 577, 579, 580, 581, 582, 583, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 703, 704, 705, 706, 711, 713, 714, 715, 716, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 734, 735, 736, 737, 738, 740, 741, 742, 743, 746, 750, 751, 752, 753, 754, 755, 759, 760, 761, 764, 765, 766, 767, 768, 769, 770, 772, 773, 774, 775, 778, 779, 780, 781, 782, 783, 785, 786, 787, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 814, 815, 816, 817, 818, 819, 820, 821, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 842, 843, 844, 846, 847, 848, 849, 850, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 881, 882, 883, 884, 885, 890, 891, 892, 894, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 939, 940, 941, 942, 944, 945, 946, 948, 951, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 970, 971, 972, 975, 976, 977, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1032, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1096, 1097, 1098, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1117, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1138, 1141, 1142, 1145, 1146, 1147, 1154, 1157, 1158, 1159, 1160, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1181, 1182, 1183, 1184, 1185, 1186, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1277, 1280, 1281, 1282, 1283, 1296, 1297, 1298, 1300, 1301, 1302, 1303, 1304, 1305, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1326, 1327, 1328, 1329, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1340, 1346, 1347, 1348, 1349, 1350, 1351, 1353, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 1369, 1370, 1371, 1372, 1373, 1376, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1387, 1388, 1389, 1390, 1393, 1394, 1396, 1397, 1398, 1399, 1400, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1429, 1430, 1431, 1433, 1434, 1435, 1436, 1439, 1440, 1441, 1442, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1453, 1454, 1455, 1457, 1458, 1459, 1460, 1461, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1492, 1493, 1494, 1496, 1497, 1498, 1499, 1500, 1505, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1520, 1524, 1525, 1526, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1605, 1606, 1607, 1608, 1611, 1612, 1613, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1633, 1634, 1635, 1636, 1637, 1638, 1641, 1642, 1643, 1644, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1666, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1686, 1689, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1724, 1727, 1728, 1729, 1730, 1733, 1734, 1735, 1736, 1749, 1750, 1751, 1754, 1755, 1756, 1757, 1758, 1759, 1762, 1763, 1764, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1795, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1835, 1838, 1848, 1849, 1850, 1867, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1894, 1895]

CNN = 'CNN'
GNN = 'GAT'
Residual = 'NO'  #change accordingly

df_e = pd.read_csv(r'C:\Users\Ahin\Desktop\insider threat\sample _data\1-data-test-undirected_edge.csv')
edge_index = torch.from_numpy(df_e.to_numpy().T)

all_result = './detection_result_transformer'
result_dir = os.path.join(all_result, 'ResHybnet_compare_sequence_length_10round')
os.makedirs(result_dir, exist_ok=True)
label_column_index=5

df_data = pd.read_csv(r'C:\Users\Ahin\Desktop\insider threat\bert_feature_extraction_result\new_dataset_bert.csv',low_memory=False)
label_column_name = df_data.columns[label_column_index]
df_data.rename(columns={label_column_name: 'label'}, inplace=True)
print("Available columns:", df_data.columns)
#selecting only the helly points
df_data['id'] = pd.to_numeric(df_data['id'], errors='coerce').fillna(0).astype(int)
df_data = df_data[df_data['id'].isin(isolated_points_list)]
#print("Labels of selected points:")
print(df_data[['id', 'label']].to_string(index=False))
#label_counts = df_data['label'].value_counts()
#print("Count of each label:")
#print(label_counts)
print("Remaining data points:", len(df_data))

for col in df_data.columns[6:]:
    df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
df_data.fillna(0, inplace=True)


feature_columns = df_data.columns[6:]
x = torch.tensor(df_data[feature_columns].values, dtype=torch.float32)
df_data['label'] = pd.to_numeric(df_data['label'], errors='coerce').fillna(0).astype(int)
y = torch.tensor(df_data['label'].values, dtype=torch.long)
print(df_data['label'].value_counts())
data = Data(x=x, y=y)


num_nodes = data.x.shape[0]
valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)

# Apply the mask
edge_index = edge_index[:, valid_mask]


data = Data(x=x, edge_index=edge_index, y=y)

# Split for training/testing
train_size = int(0.8 * data.num_nodes)
indices = torch.randperm(data.num_nodes)
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[indices[:train_size]] = 1
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask[indices[train_size:]] = 1

print("edge_index shape:", edge_index.shape)  
print("Filtered edge count:", edge_index.shape[1])

#just to ensure both labels are present
count_0 = df_data['label'].eq(0).sum()
count_1 = df_data['label'].eq(1).sum()

print(count_0, count_1)
data.train_mask = train_mask
data.test_mask = test_mask
# Defined lists to store metrics for each round
results=[]
round_train_accuracies = []
round_test_accuracies = []
round_train_losses = []
round_test_losses = []
round_train_f1 = []
round_test_f1 = []
round_train_precision = []
round_test_precision = []
round_train_recall = []
round_test_recall = []

for r in trange(train_round):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResHybNet(
        input_dim=data.num_node_features,
        output_dim=data.num_node_features,
        cnn=CNN,
        gnn=GNN,
        residual=Residual
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-4)

    best_model_path = os.path.join(result_dir, "early_stop_model")
    os.makedirs(best_model_path, exist_ok=True)
    best_model_path = os.path.join(best_model_path, f'{CNN}_{GNN}_{Residual}_{r}round_best.pt')

    early_stopping = EarlyStopping(save_path=best_model_path, verbose=True, patience=15, delta=0.0001, metric='loss')
    all_train_acc = []
    all_test_acc = []
    all_train_loss = []
    all_test_loss = []
    all_train_f1 = []
    all_test_f1 = []
    all_train_precision = []
    all_test_precision = []
    all_train_recall = []
    all_test_recall = []
    f1_scores=[]
    for epoch in range(EPOCH):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        out_prob = F.softmax(out, dim=1)
        threshold = 0.4
        pred_y = (out_prob[:, 1] >= threshold).long()

        # Training metrics
        pred_y_train = torch.masked_select(pred_y, data.train_mask.bool()).tolist()
        true_y_train = data.y[data.train_mask.bool()].tolist()
        train_acc_s = accuracy_score(true_y_train, pred_y_train)
        train_f1 = f1_score(true_y_train, pred_y_train, average='binary')
        train_precision = precision_score(true_y_train, pred_y_train, average='binary')
        train_recall = recall_score(true_y_train, pred_y_train, average='binary')

        loss = F.cross_entropy(out[data.train_mask.bool()], data.y[data.train_mask.bool()].long())
        loss.backward()
        optimizer.step()
        train_loss_s = loss.item()

        model.eval()
        out = model(data)
        out_prob = F.softmax(out, dim=1)
        pred_y_test = (out_prob[:, 1] >= threshold).long()

        # Test metrics
        pred_y_test = torch.masked_select(pred_y_test, data.test_mask.bool()).tolist()
        true_y_test = data.y[data.test_mask.bool()].tolist()
        test_acc = accuracy_score(true_y_test, pred_y_test)
        test_f1 = f1_score(true_y_test, pred_y_test, average='binary')
        test_precision = precision_score(true_y_test, pred_y_test, average='binary')
        test_recall = recall_score(true_y_test, pred_y_test, average='binary')

        test_loss = F.cross_entropy(out[data.test_mask.bool()], data.y[data.test_mask.bool()].long())

        # Store metrics
        all_train_acc.append(train_acc_s)
        all_test_acc.append(test_acc)
        all_train_loss.append(train_loss_s)
        all_test_loss.append(test_loss.item())
        all_train_f1.append(train_f1)
        all_test_f1.append(test_f1)
        all_train_precision.append(train_precision)
        all_test_precision.append(test_precision)
        all_train_recall.append(train_recall)
        all_test_recall.append(test_recall)

        # Early stopping
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            break

    # Store round results
    round_train_accuracies.append(all_train_acc)
    round_test_accuracies.append(all_test_acc)
    round_train_losses.append(all_train_loss)
    round_test_losses.append(all_test_loss)
    round_train_f1.append(all_train_f1)
    round_test_f1.append(all_test_f1)
    round_train_precision.append(all_train_precision)
    round_test_precision.append(all_test_precision)
    round_train_recall.append(all_train_recall)
    round_test_recall.append(all_test_recall)

    print(f"Round {r + 1} completed.")

    # Plot per-round accuracy and loss
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(all_train_acc) + 1), all_train_acc, label='Train Accuracy', color='blue')
    plt.plot(range(1, len(all_test_acc) + 1), all_test_acc, label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training vs Validation Accuracy - Round {r + 1}')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(all_train_loss) + 1), all_train_loss, label='Train Loss', color='blue')
    plt.plot(range(1, len(all_test_loss) + 1), all_test_loss, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss - Round {r + 1}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate best model
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    out = model(data)
    out_prob = F.softmax(out, dim=1)
    pred_y_final = (out_prob[:, 1] >= threshold).long()
    pred_y_final = torch.masked_select(pred_y_final, data.test_mask.bool()).tolist()
    true_y_final = data.y[data.test_mask.bool()].tolist()

    accuracy = accuracy_score(true_y_final, pred_y_final)
    precision = precision_score(true_y_final, pred_y_final, zero_division=0)
    recall = recall_score(true_y_final, pred_y_final, zero_division=0)
    f1 = f1_score(true_y_final, pred_y_final, average='weighted')

    results.append([r + 1, accuracy, precision, recall, f1, epoch + 1])

    # Store per-round metrics
    round_train_accuracies.append(all_train_acc[-1])
    round_test_accuracies.append(all_test_acc[-1])
    round_train_losses.append(all_train_loss[-1])
    round_test_losses.append(all_test_loss[-1])
    f1_scores.append(f1)

# ROC Curve
plt.figure(figsize=(6, 6))
fpr, tpr, _ = roc_curve(true_y_final, pred_y_final)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()



# Save results
df_results = pd.DataFrame(results, columns=["Round", "Accuracy", "Precision", "Recall", "F1-Score", "Epochs"])
output_file = os.path.join(r'C:\Users\Ahin\Desktop\insider threat\bert_feature_extraction_result\accuracyno_new.xlsx')
df_results.to_excel(output_file, index=False)

print('F1 score:', f1)
print(f"Results saved to {output_file}")

