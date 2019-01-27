import mrcnn.model as modellib
from satellite.satellite import SatelliteDataset, SatelliteConfig
from skimage.measure import find_contours, approximate_polygon
import cv2
import math
import numpy as np

class MaskRCNN():
    def __init__(self, 
                 modelPath, 
                 instance_number_threshold, 
                 area_set
                 ):
        self.modelPath = modelPath

        ##### rewrite after this line
        config = SatelliteConfig()
        config.display()
        self.dataset_val = SatelliteDataset()
        self.dataset_val.config_dataset(dataset_name='satellite',
                                   json_file='../data/complete.json',
                                   skip_classes=['Road','Body_Of_water','Tree'],
                                   root_dir='./data',
                                   instance_number_threshold=instance_number_threshold,
                                   area_set=area_set
                                   )

        self.dataset_val.prepare()
        class InferenceConfig(SatelliteConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        inference_config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference',
                                  config=inference_config,
                                  model_dir='./tmp')

        # model_path = "./checkpoint/mask_rcnn_satellite_0005.h5"
        self.model.load_weights(self.modelPath, by_name=True)

    def infere(self, image, imageId=None, debug=False):
            data = self.model.detect([image], verbose=debug)[0]
            result = []

            for i in range(data['scores'].shape[0]):

                label = self.dataset_val.class_names[data['class_ids'][i]]
                mask = data['masks'][:, :, i]
                mask = cv2.resize(mask.astype(np.uint8), (1024,1024))
                area, perimetr, cv2Poly   = self.getMaskInfo(mask, (10,10))

                if cv2Poly is None:
                    print("Warning: Object is recognized, but contour is empty!")
                    continue

                verts = cv2Poly[:,0,:]
                r = {'classId': data['class_ids'][i],
                     'score': data['scores'][i],
                     'label': label,
                     'area': area,
                     'perimetr': perimetr,
                     'verts': verts}

                if imageId is not None:
                    r['objId'] = "{}_obj-{}".format(imageId, i)

                result.append(r)

            return result

    def getMaskInfo(self, img, kernel=(10, 10)):

        #Define kernel
        kernel = np.ones(kernel, np.uint8)

        #Open to erode small patches
        thresh = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        #Close little holes
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations=4)

        thresh=thresh.astype('uint8')
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        maxArea = 0
        maxContour = None

        # Get largest area contour
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > maxArea:
                maxArea = a
                maxContour = cnt

        if maxContour is None: return [None, None, None]

        perimeter = cv2.arcLength(maxContour,True)

        # aproximate contour with the 1% of squared perimiter accuracy
        # approx = cv2.approxPolyDP(maxContour, 0.01*math.sqrt(perimeter), True)

        return maxArea, perimeter, maxContour





