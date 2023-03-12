import json
with open('bert_chinese_BertCoGAttV3_None_seed_1err.txt','r',encoding='utf-8') as f:
	data = json.load(f)

diseases = []
with open('../Data/label2id.txt','r',encoding='utf-8') as f:
	for line in f:
		line = line.strip()
		if line == '':
			continue
		diseases.append(line.split(' ')[0])

for disease in diseases:
	err_data = []
	for item in data:
		if (disease in item['diagnosis'] and disease not in item['err_diagnosis'] ) or\
		   (disease not in item['diagnosis'] and disease in item['err_diagnosis'] ):
		    err_data.append(item)
	with open(f'{disease}.txt','w',encoding='utf-8') as f:
		json.dump(err_data,f,ensure_ascii=False,indent= 4)




