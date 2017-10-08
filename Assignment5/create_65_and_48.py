
import pandas as pd
import numpy as np


# Change paths accordingly
path = '/home/xu/Documents/Intro to Data Science/Assignment5/'
features_file = 'featuresFlowCapAnalysis2017.csv'
labels_file = 'labelsFlowCapAnalysis2017.csv'

save_dataset1 = 'train_65.csv'
save_dataset2 = 'train_48.csv'

d1 = pd.read_csv(path+features_file)


d1.columns = list(range(0,186))

#complete_data = np.delete(complete_data,47)
leave_only = ['133', '10', '132', '122', '121', '125', '137', '183', '6', '40', '36', '7', '38', '8', '182', '91', '120', '181', '1', '136', '90', '2', '95', '123', '53', '48', '87', '94', '176', '14', '37', '3', '174', '134', '180', '167', '52', '178', '124', '89', '81', '4', '85', '162', '129', '166', '5', '0', '135', '79', '185', '141', '39', '83', '50', '99', '93', '26', '68', '49', '168', '27', '117', '16', '57']
leave_only_int = [int(n) for n in leave_only]
complete_data = d1[leave_only_int]

leave_only_int1 = leave_only_int[:48]
df2 = d1[leave_only_int1]

print(np.count_nonzero(np.isnan(complete_data)))

complete_data.to_csv(path+save_dataset1)
df2.to_csv(path+save_dataset2)