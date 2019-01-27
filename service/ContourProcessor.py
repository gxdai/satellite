import os
import numpy as np
import osgeo.ogr as ogr
import osgeo.osr as osr
import geopandas as gpd

class ContourProcessor():
    def __init__(self, geoImg, outputDir):

        self.geoImg = geoImg

        # If re projection is needed
        # latlng = osr.SpatialReference()
        # latlng.ImportFromEPSG(4326)
        # self.projTransform = osr.CoordinateTransformation(self.geoImg.tifProjection, latlng)

        driver = ogr.GetDriverByName("GeoJSON")
        self.fullDataGeojsonPath = os.path.join(outputDir, 'fullData.geojson')
        self.fullDataSource = driver.CreateDataSource(self.fullDataGeojsonPath)
        # create the layer
        self.fullDataLayer = self.fullDataSource.CreateLayer("payload", self.geoImg.tifProjection, ogr.wkbPolygon)
        # Add the fields we're interested in
        field_type = ogr.FieldDefn("Label", ogr.OFTString)
        field_type.SetWidth(24)
        self.fullDataLayer.CreateField(field_type)
        field_type = ogr.FieldDefn("ObjId", ogr.OFTString)
        field_type.SetWidth(24)
        self.fullDataLayer.CreateField(field_type)
        self.fullDataLayer.CreateField(ogr.FieldDefn("AreaSqPx", ogr.OFTInteger))
        self.fullDataLayer.CreateField(ogr.FieldDefn("ClassId", ogr.OFTInteger))
        self.fullDataLayer.CreateField(ogr.FieldDefn("Score", ogr.OFTReal))

        self.cleanedGeojsonPath = os.path.join(outputDir, 'cleanData.geojson')

        self.patchesGeojsonPath = os.path.join(outputDir, 'patchesData.geojson')
        self.patchesSource = driver.CreateDataSource(self.patchesGeojsonPath)
        # create the layer
        self.patchesLayer = self.patchesSource.CreateLayer("payload", self.geoImg.tifProjection, ogr.wkbPolygon)

    def addPatchBoundary(self, left, up):
        bbox = np.array([[0,0], [0,1023], [1023,1023], [1023,0]], np.float)
        feature = ogr.Feature(self.patchesLayer.GetLayerDefn())
        bbox[::,::] += (float(left), float(up))
        feature.SetGeometry(self.reproject(bbox))
        self.patchesLayer.CreateFeature(feature)

    def addFeature(self, left, up, modelInference):
        score = modelInference['score']
        classId = modelInference['classId']
        label = modelInference['label']
        polyVerts = modelInference['verts']

        feature = ogr.Feature(self.fullDataLayer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField("Label", label)
        feature.SetField("ClassId", int(classId))
        feature.SetField("Score", float(score))
        feature.SetField("ObjId", modelInference['objId'])
        feature.SetField("AreaSqPx", modelInference['area'])
        polyVerts[::,::] += (int(left), int(up))
        feature.SetGeometry(self.reproject(polyVerts))
        # Create the feature in the layer (geojson)
        self.fullDataLayer.CreateFeature(feature)
        # Dereference the feature
        feature = None

    def cleanUp(self):
        # save and close patch and fulldata shape files
        self.fullDataSource = None
        self.patchesSource = None

        #load full data file in a dataframe
        gdf = gpd.GeoDataFrame.from_file(self.fullDataGeojsonPath)

        intersectGroups = gpd.sjoin(gdf, gdf, how="inner", op='intersects').groupby('ObjId_left')

        gdf.set_index(['ObjId'], inplace=True)
        gdf.sort_values(by=['Score'], ascending=False, inplace=True)

        itemsToRemove = {}

        print(len(gdf))

        for objId in gdf.index:
            if objId in itemsToRemove: continue
            idf = intersectGroups.get_group(objId)
            objToRemove = idf[idf['ObjId_left'] != idf['ObjId_right']][['ObjId_right']].values
            for id in objToRemove: itemsToRemove[id[0]] = True

        gdf.drop(itemsToRemove.keys(), inplace=True)

        gdf.to_file(driver='GeoJSON', filename=self.cleanedGeojsonPath)

    def reproject(self, relativePolygon):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        xoffset, px_w, rot1, yoffset, rot2, px_h = self.geoImg.afineTrnsform

        for x,y in relativePolygon:
            posX = px_w * x + rot1 * y + xoffset
            posY = rot2 * x + px_h * y + yoffset

            # shift to the center of the pixel
            posX += px_w / 2.0
            posY += px_h / 2.0

            ring.AddPoint_2D(posX, posY)

        x, y = ring.GetPoint_2D(0)
        ring.AddPoint_2D(x, y)
        # ring.Transform(self.projTransform)

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # print(poly.ExportToWkt())

        return poly.Simplify(0.5)

