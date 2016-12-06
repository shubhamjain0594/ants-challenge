import glob
import os
import scipy.ndimage
import numpy as np
import csv,sys

DATA_PATH = '/data4/ants-challenge'
train_valid_split = 0.8

files = glob.glob(os.path.join(DATA_PATH,'fnames/*.jpeg'))
train_images = files[:int(train_valid_split*len(files))]
valid_images = files[int(train_valid_split*len(files)):]

def load_csv(file_name):
    """
    Reads CSV at file_name as list of lists for each line.
    """
    if sys.version_info[0] < 3:
        lines = []
        infile = open(file_name, 'rb')
    else:
        lines = []
        infile = open(file_name, 'r', newline='')

    ant_dict = {}
    with infile as f:
        csvreader = csv.reader(f)
        for c,lines in enumerate(csvreader):
            if c==0:
                continue
            ant_id = lines[0]
            if ant_id not in ant_dict:
                ant_dict[ant_id] = {}
                ant_dict[ant_id]['frames'] = []
                ant_dict[ant_id]['coordiantes'] = []
            if lines[-1]:
                ant_dict[ant_id]['frames'].append('{:05d}'.format(int(lines[1])))
                ant_dict[ant_id]['coordiantes'].append([lines[2], lines[3]])

        return ant_dict




def data_generator(mode,ant_id,batch_size=2,xs =288,ys=512,s1=270,s2=480):
    ant_dict = load_csv(os.path.join(DATA_PATH,'training_dataset.csv'))
    spc_ant_dict = ant_dict[ant_id]

    frames = spc_ant_dict['frames']
    frames = [DATA_PATH+'/frames/'+x+'.jpeg' for x in frames]

    if mode=='train':
        images = frames[:int(len(frames)*train_valid_split)]
    elif mode == 'valid':
        images = frames[int(len(frames)*train_valid_split):]

    for index in range(len(images)):
        X = None
        Y1 = None
        Y2 = None
        for file_index,files in enumerate(images[index:index+batch_size]):
            img = scipy.ndimage.imread(files)
            coord = (float(spc_ant_dict['coordiantes'][index + file_index][0]),
                        float(spc_ant_dict['coordiantes'][index + file_index][1]))
            for i in range(0,img.shape[0],s1):
                for j in range(0,img.shape[1],s2):
                    if i+xs <= img.shape[0] and j+ys <= img.shape[1]:
                        crop_img = img[i:i+xs,j:j+ys,:]
                    elif i+xs > img.shape[0] and j+ys <= img.shape[1]:
                        crop_img = img[-xs:,j:j+ys,:]
                    elif j+ys > img.shape[0]and i+xs <= img.shape[0]:
                        crop_img = img[i:i+xs:,-ys:,:]
                    else:
                        crop_img = img[-xs:,-ys:,:]
                    if i <= coord[1] < i+xs and j <= coord[0] <j+ys:
                        y1 = np.asarray([0,1])
                        y2 = np.asarray([coord[0], coord[1]])
                        y1 = y1.reshape((1,2))
                        y2 = y2.reshape((1,2))
                    else:
                        y1 = np.asarray([1,0])
                        y2 = np.asarray([0, 0])
                        y1 = y1.reshape((1,2))
                        y2 = y2.reshape((1,2))

                    crop_img = crop_img.reshape((1,xs,ys,3))

                    if X is None:
                        X = crop_img
                        Y1 = y1
                        Y2 = y2
                    else:
                        X = np.concatenate((X,crop_img),axis =0)
                        Y1 = np.concatenate((Y1,y1),axis =0)
                        Y2 = np.concatenate((Y2,y2),axis =0)
        yield X,Y1,Y2








if __name__ == '__main__':
    count = 0
    for X,Y1,Y2 in data_generator(mode = 'train', ant_id = '101'):
        print (X.shape)
        print (Y1.shape)
        print (Y2.shape)
        if count == 0:
            break
