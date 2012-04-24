
import os

def generate_val_train_msrc(directory,random_init):
    import random
    with open(os.path.join(directory,'images.txt')) as f:
        image_names = f.readlines()
    class_images = [[] for i in xrange(1,9)]
    for image_name in image_names:
        class_images[int(image_name[0])-1].append(image_name)
    with open(os.path.join(directory,'train_%d.txt'%random_init),'w') as train_list:
        with open(os.path.join(directory,'val_%d.txt'%random_init),'w') as val_list:
            for class_list in class_images:
                random.shuffle(class_list)
                train_list.writelines(class_list[:len(class_list)/2])
                val_list.writelines(class_list[len(class_list)/2:])
# MSRC
colors=np.array([[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128], [192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,64,128],[0,192,128],[128,192,128],[64,64,0],[192,64,0]])
classes=['building','grass','tree','cow','horse','sheep','sky','mountain','aeroplane','water','face','car','bicycle','flower','sign','bird','book','chair','road','cat','dog','body','boat']
convert=[99./1000,587./1000,114./1000]
label_dict=np.dot(colors,convert).tolist()
