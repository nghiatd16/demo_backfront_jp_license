import cv2
import numpy as np
def me_median(a):
    return np.mean(a, axis = 0)
def connected_CC(stat1,stat2):
    x_new = min(stat1[0],stat2[0])
    y_new = min(stat1[1],stat2[1])
    w_new = max(stat1[0]+stat1[2],stat2[0]+stat2[2])-x_new
    h_new = max(stat1[1]+stat1[3],stat2[1]+stat2[3])-y_new
    area_new = stat1[4]+stat2[4]
    return [x_new,y_new,w_new,h_new,area_new]
    

def merge_CC(stats_final_):
    #[x y w h area]
    # kiem tra CC chong cheo nhau truoc
    stats_list = stats_final_.copy()
    i = 0
    while(i<len(stats_list)-1):
        if stats_list[i][0]<=stats_list[i+1][0] and  stats_list[i][0]+stats_list[i][2]>=stats_list[i+1][0]:
            #dis_ = stats_list[i][0]+stats_list[i][2] - stats_list[i+1][0]
            #if dis_>0.5*stats_list[i][2] or dis_>0.5*stats_list[i+1][2]:
                stats_list[i] =  connected_CC(stats_list[i],stats_list[i+1])    
                stats_list = np.delete(stats_list,i+1,axis=0)
            #else: i+=1
        else:i=i+1
        # co chong cheo xay ra
    me_median_width = me_median(stats_list[:,2])
    i = 0
    while(i<len(stats_list)):
        # kiem tra noi neu < median_width
        if(stats_list[i][2]< int(0.61*me_median_width)):
            if(i==0):# noi voi cai tiep theo
                stats_list[i+1] = connected_CC(stats_list[i],stats_list[i+1])
                stats_list = np.delete(stats_list,i,axis=0)
                continue
            elif i==len(stats_list)-1: # noi voi cai truoc do
                stats_list[i-1] = connected_CC(stats_list[i-1],stats_list[i])
                stats_list = np.delete(stats_list,i,axis=0)
                continue
            else: #kiem tra khoang cac den 2 cai gan nhat
                if(stats_list[i][0]-stats_list[i-1][0]-stats_list[i-1][2]
                  <stats_list[i+1][0]-stats_list[i][0]-stats_list[i][2]):
                    
                    if(stats_list[i][2]*0.6>stats_list[i][0]-stats_list[i-1][0]-stats_list[i-1][2]):
                        stats_list[i-1] = connected_CC(stats_list[i-1],stats_list[i])
                        stats_list = np.delete(stats_list,i,axis=0)
                        continue
                elif (stats_list[i][2]*0.6>stats_list[i+1][0]-stats_list[i][0]-stats_list[i][2]):
                    stats_list[i+1] = connected_CC(stats_list[i],stats_list[i+1])
                    stats_list = np.delete(stats_list,i,axis=0)
                    continue

            
        i=i+1    
    return stats_list
def pos_processing(processed_area,stats):
    stats_list = stats.copy()
    # tinh lại các meandian
    me_median_width = me_median(stats_list[:,2])
    me_median_heigh = me_median(stats_list[:,3])
    me_meadian_raito = me_median(np.array(stats_list[:,2])/np.array(stats_list[:,3]))
    
    me_median_y = me_median(stats_list[:,1])
    ###########################
    #loai bo 1 so noise có size lon ma preprocessing khong loai duoc
    i=0
    while(i<len(stats_list)):
        #xoa những noise có size khá lớn phía bên dưới
        if stats_list[i][1]>me_median_heigh:
            stats_list = np.delete(stats_list,i,axis=0)
        elif stats_list[i][1]<me_median_y and  stats_list[i][3]<me_median_heigh*0.6: 
            # xoa nhưng noi có size khá lớn phía bên trên
            stats_list = np.delete(stats_list,i,axis=0)
        else: i=i+1
            #
    #############################################################################
    #kiem tra ki tu thieu net o tren hoac duoi do quá trình preprocessing
    # mo rong h theo phia duoi, hoac thay doi (x,y) ve phia tren dua vao mean y
    i = 0
    while(i<len(stats_list)-1):
        # thieu net o phia tren
        if stats_list[i][1] > me_median_y \
        and stats_list[i][3]>=0.4*me_median_heigh \
        and stats_list[i][3]<0.8*me_median_heigh: # loai bo truong hop dau '-' va noise 
            #noi len tren
            # tinh khoang can noi
            l = 1.1*me_median_heigh - stats_list[i][3]
            stats_list[i][1] = stats_list[i][1]-l if stats_list[i][1]-l>0 else 0
            stats_list[i][3] = 1.1*me_median_heigh
        #thieu net o phia duoi
        if stats_list[i][1] <me_median_y \
        and stats_list[i][3]>=0.3*me_median_heigh \
        and stats_list[i][3]<0.75*me_median_heigh: # loai bo truong hop dau '-' 
            #noi xuoi duoi
            stats_list[i][3] = 1.1*me_median_heigh
        i=i+1

    #####################################################################
    #kiem tra 2 ki tu dinh lien voi nhau
    #####################################################################
    i=0
    while(i<len(stats_list)-1):
        # thieu net o phia tren
        if stats_list[i][2]/stats_list[i][3] > 1.5*me_meadian_raito \
        and np.abs(1-stats_list[i][3]/me_median_heigh)<=0.3:
            x,y,w,h,are = stats_list[i]
            img_process = processed_area[y:y+h,x:x+w]
            # chia lại
            img = np.array(img_process)
            img = 255-img
            address_area =cv2.medianBlur(img,3)    
            
            proces = cv2.adaptiveThreshold(address_area, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                           151, 2)
            kernel = np.ones((4,2),np.uint8)
            erosion = cv2.erode(proces,kernel,iterations = 2)
            
            num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(erosion,4, cv2.CV_32S)
            
            labels_in_mask = list(np.arange(1,num_labels))
            arr_ret = np.array([stats[s] for s in labels_in_mask])
            arr_temp = np.argsort(arr_ret[:,0])
            stats = np.array([arr_ret[s] for s in arr_temp])


            for it in stats:
                it[0]+=stats_list[i][0]
                it[1]+=stats_list[i][1]
            j = 0
            while(j<len(stats)-1):
                if stats[j][0]<=stats[j+1][0] and  stats[j][0]+stats[j][2]> stats[j+1][0]:
                        s_ =  connected_CC(stats[j],stats[j+1])  
                        stats[j]=s_
                        stats = np.delete(stats,j+1,axis=0)
                else:j=j+1   
            
            
            stats_list = np.delete(stats_list,i,axis=0)
            stats_list = list(stats_list)
            for it in stats:
                if it[4]>300:
                    stats_list.append(it)

        i=i+1    
    return stats_list
    
def Final_Binary_convert(img,connectivity=4):
    address_area =cv2.medianBlur(img,3)    

    processed_area = cv2.adaptiveThreshold(address_area, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                           151, 2)
    #_,processed_area = cv2.threshold(address_area,75,255,cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #kernel = np.ones((3,3),np.uint8)
    processed_area = cv2.morphologyEx(processed_area, op=cv2.MORPH_OPEN, kernel=kernel, anchor=(-1, -1), iterations=2)
    #CC
    num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(processed_area, connectivity, cv2.CV_32S)
    
    mask = np.ones_like(labels)

    labels_in_mask = list(np.arange(1,num_labels))

    areas = [s[4] for s in stats]

    sorted_idx = np.argsort(areas)

    areas_final = [areas[s] for s in sorted_idx][:-1]
    
    for lidx, cc in zip(sorted_idx, areas_final):
        if cc <= 125:
            mask[labels == lidx] = 0
            labels_in_mask.remove(lidx)
    
    x=stats[:,0][1:]
    y=stats[:,1][1:]
    height,width = img.shape
    
    idx = np.arange(0,len(x)+1)[1:]
    for i,x_ in enumerate(x):
        if y[i]>=0.86500*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue
        # xoa nhung vạch nhỏ sát viền trên
        if y[i]<=(1-0.96)*height and stats[i+1][3]<=(1-0.96)*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue
        #xóa những vạch nhỏ sát mép phải
        if x[i]>=(1-0.996)*width and stats[i+1][2]<=(1-0.996)*width:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue
        # xoa những vạch kẻ tương đối lớn
        if stats[i+1][2]>=0.6*width or  stats[i+1][2] > 3*height:
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue
        # xoa nhung net thang dung(ap dung cho 1 line)
        if stats[i+1][3]>=0.96*height :#and stats[i+1][3]<0.5*height 
            mask[labels == (i+1)] = 0
            if i+1 in labels_in_mask:labels_in_mask.remove(i+1)
            continue 
    if len(labels_in_mask) == 0:
        return processed_area, []
    arr_ret = np.array([stats[s] for s in labels_in_mask])
    # print(arr_ret)
    arr_temp = np.argsort(arr_ret[:,0])
    stats_final = np.array([arr_ret[s] for s in arr_temp])
    stats_all = merge_CC(stats_final)  
    #pos-processing
    stats_all = pos_processing(img,stats_all)
    processed_area = processed_area*mask

    processed_area = 255 - processed_area
    stats_all = sorted(stats_all, key=lambda tup:tup[0])
    return processed_area,stats_all