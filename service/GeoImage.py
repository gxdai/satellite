import numpy as np
import cv2
import copy
from osgeo import gdal, osr, ogr

class GeoImage():
    def __init__(self,
                 imgPath,
                 gap=100,
                 subsize=1024):
        self.imgPath = imgPath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.imgPath = imgPath
        self.splits = []
        self.BuildSplits(1)

        ds = gdal.Open(imgPath)
        self.afineTrnsform = ds.GetGeoTransform()
        self.tifProjection = osr.SpatialReference(wkt=ds.GetProjection())

        del ds

    def getProjection(self):
        return self.tifProjection

    def getCv2ImgFromSplit(self, xyOffset):
        left, up = xyOffset
        subimg = copy.deepcopy(self.resizeimg[up: (up + self.subsize), left: (left + self.subsize)])
        return subimg

    def BuildSplits(self, rate):
        img = cv2.imread(self.imgPath)
        # print("img.shape = {}".format(img.shape))
        # convert image from bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert np.shape(img) != ()

        if (rate != 1):
            self.resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            self.resizeimg = img

        weight = np.shape(self.resizeimg)[1]
        height = np.shape(self.resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)

                self.splits.append([left, up])

                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def getSplits(self): return self.splits


if __name__ == '__main__':
    geoImg = GeoImage(r'/home/kirill/Downloads/GeoData/Davis_Monthan/Davis_Monthan_AFB_20180814.tif')
    print(geoImg.getSplits())
