import numpy as np
import operator

def read_data_from_set(src_file, dest_file, id_file):

    with open(id_file, 'r') as ques_id_file:
        ques_id = [id.rstrip('\n') for id in ques_id_file]
        
    with open("distinctquestionid.txt") as distinct_id_file:
        distinct_id = [id.rstrip('\n') for id in distinct_id_file]

    with open(src_file, 'r') as pred_val_file:
        pred_val = [id.rstrip('\n') for id in pred_val_file]

    with open(dest_file, 'r') as out_val_file:
        out_val = [id.rstrip('\n') for id in out_val_file]

    five_nearest_neighbor = [[0 for x in range(30)] for x in range(5)]
    pred_data = [0 for x in range(30)]
    out_data = [0 for x in range(30)]
    dist_map = {}
    """cosine_map = {}"""
    ques_index = 0
    pred_index = 0
    out_index = 0
    position = 0
    fail = 0
    win = 0
    for pred_index, pred_element in enumerate(pred_val):
        pred_results = pred_element.replace("\n","").replace("[","").replace("]","").split(',')
        pred_data = [float(i) for i in pred_results]

        for out_index, out_element in enumerate(out_val):
            out_results = out_element.replace("\n","").replace("[","").replace("]","").split(',')
            out_data = [float(j) for j in out_results]
            dist = np.linalg.norm(np.asarray(out_data) - np.asarray(pred_data))
            dist_map[distinct_id[out_index]] = dist
            """cosine = np.dot(np.mat(out_data), np.mat(pred_data).T)/np.linalg.norm(out_data)/np.linalg.norm(pred_data) 
            cosine_map[distinct_id[out_index]] = cosine"""

        result_list = sorted(dist_map.iteritems(), key=operator.itemgetter(1), reverse=False)
        
        for x, y in enumerate(result_list):
            if y[0] == ques_id[ques_index]:
                position = x
                
        print str(ques_id[ques_index]) +" is in "+str(position)
        
        ques_index += 1
        if position > 4:
            fail += 1
        else:
            win += 1
    print "fail = "+str(fail)+" win = "+str(win)

if __name__ == "__main__":
    read_data_from_set("testtest.txt", "vector.txt", "id.txt")
