import pandas as pd
import numpy as np
mutations = pd.read_csv('../project_2_data/TCGA_HNSC_mutations_cleaned.txt', sep = '\t')
by_patient = mutations.groupby(by = ['Chromosome', 'Start_Position', 'End_Position']).agg({'Variant_Classification' : [lambda x: np.count_nonzero(x == 'Silent') > 0, lambda x: np.count_nonzero(x == 'Missense_Mutation') > 0], 'Reference_Allele' : 'first', 'Tumor_Seq_Allele2' : 'first', 'patient_id' : list}).rename(columns = {'<lambda_0>' : 'Silent', '<lambda_1>' : 'Missense_Mutation'})
by_patient.columns = by_patient.columns.map(lambda x: x[1] if x[0] == 'Variant_Classification' else x[0])
by_patient = by_patient.reset_index()
by_patient = by_patient[['patient_id', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2']].rename(columns = {'patient_id' : 'sampleID', 'Chromosome' : 'chr', 'Start_Position' : 'pos', 'Reference_Allele' : 'ref', 'Tumor_Seq_Allele2' : 'mut'})
by_patient.to_csv('./dndscv_inp.csv', index = False)