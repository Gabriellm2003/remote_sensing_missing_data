import argparse



def main():
    parser = argparse.ArgumentParser(description='Log parser to select the best model.')
    parser.add_argument('--log_path', type=str, required=True,
                        help='Path to log file.')
    args = parser.parse_args()
    log_path = args.log_path

    f = open(log_path, 'r')
    
    grd_dict = {}
    aer_dict = {}
    key = ''

    while True:
    	
    	for i in range(22):
    		line = f.readline() 
    		if (line == ''):
    			break
    		if (i == 0):
    			if (line.split(' ')[1].replace('\n','') not in grd_dict.keys()):
    				key = line.split(' ')[1].replace('\n','')
    				grd_dict [key] = []
    				aer_dict [key] = []
    		if (i in [5, 6, 7, 8, 9, 10, 11, 12]):
    			mAP = float(line.split(': ')[1])
    			grd_dict [key].append(mAP)
    		if (i in [14, 15, 16, 17, 18, 19, 20, 21]):
    			mAP = float(line.split(': ')[1])
    			aer_dict [key].append(mAP)  
    	if (line == ''):
    		break
    
    mean_1 = []
    mean_2 = []
    for k, v in grd_dict.items():
    	list_sum = 0
    	for i in v:
    		list_sum += i
    	mean_1.append(list_sum/8)

    for k, v in aer_dict.items():
    	list_sum = 0
    	for i in v:
    		list_sum += i
    	mean_2.append(list_sum/8)

    print ("Best model for ground queries: ")
    print (mean_1.index(max(mean_1)))
    print ("Best model for aerial queries: ")
    print (mean_2.index(max(mean_2)))
    

if __name__ == '__main__':
    main()