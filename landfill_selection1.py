#Developed by Md Touhidur Rahman (www.linkedin.com/in/md-touhidur-rahman)
#This models aims to find suitable sites for landfill. It takes into consideration the following criteria: 
#Land Cover (vegetation, bare land, buildings), Land Slope and Distance from Roads & Streams

import arcpy
import os
import time
import rasterio
import matplotlib.pyplot as plt
import ee
import geemap
import folium
from folium import plugins
import geopandas as gpd
from IPython.display import display


arcpy.env.overwriteOutput = True


#SET VARIABLES BEFORE RUNNING
project_path = r"E:\OneDrive - VEI\portfolio_projects\landfill_selection"
geodatabase_name = "geodatabase3"
output_path = r"E:\OneDrive - VEI\portfolio_projects\landfill_selection\output_files"
aoi_path = r"E:\OneDrive - VEI\portfolio_projects\landfill_selection\input_files\dhaka_city.shp"
roads_path = r"E:\OneDrive - VEI\portfolio_projects\landfill_selection\input_files\dhaka_roads.shp"

#DEFINE TIME PERIOD FOR RASTER DATA
start_date = '2022-01-01'
end_date = '2022-12-31'

#THE FOLLOWING FOLDER MUST BE SYNCED WITH YOUR LOCAL SYSTEM. AFTER AUTHENTICATION, EXTRACTED NDVI, BARE SOIL, BUILDINGS,
##DEM WILL BE SAVED IN THE GOOGLE DRIVE. THESE FILES ARE MEANT TO BE ON THE LOCAL SYSTEM THROUGH SYNC

#GOOGLE DRIVE OUTPUT DIRECTORY NAME
output_dir = 'GEE'

#GOOGLE DRIVE SYNC PATH
google_drive_path = r"G:\My Drive\GEE"

#TIME (IN SECONDS) REQUIRED TO SYNC DATA FROM GOOGLE DRIVE TO LOCAL DRIVE
waiting_time = 300

#CHANGE THE WEIGHTAGE IF NEEDED
road_weight = 20
stream_weight = 15
slope_weight = 15
vegetation_weight = 10
buildings_weight = 20
bare_soil_weight = 20


#WEIGHT MATRIX FOR THE FIVE CRITERIA. HIGHER VALUE INDICATED HIGHER WEIGHTAGE/SUITABLE SITES
road_matrix = [1, 1, 2, 4, 4]
stream_matrix = [2, 2, 2, 3, 4]
slope_matrix = [5, 4, 3, 2, 1]
vegetation_matrix = [1, 2, 4, 4, 2]
buildings_matrix = [1, 2, 3, 4, 5]
bare_soil_matrix = [3, 3, 4, 4, 5]


# AUTHENTICATE EARTH ENGINE
print("Connecting to Google Earth Engine. Paste your authentication code.")
ee.Authenticate()
ee.Initialize()

start_time = time.time()

#CREATE A GEODATABASE
try:
    arcpy.management.CreateFileGDB(f"{project_path}", f"{geodatabase_name}", "CURRENT")
    geodatabase_path = f"{project_path}\\{geodatabase_name}.gdb"
    print(geodatabase_path)
    print(f"{geodatabase_name} created")
except:
    print("Failed to create geodatabase")


#FUNCTION TO SAVE TIFF TO A FOLDER
def save_tif(input_path, output_name):
    arcpy.management.CopyRaster(input_path, f"{output_path}//{output_name}.tif", '', None, "0", 
        "NONE", "NONE", '', "NONE", "NONE", "TIFF", "NONE", "CURRENT_SLICE", "NO_TRANSPOSE")


#FUNCTION TO DISPLAY RASTER USING RASTERIO
def display_raster(file_name, color):
    path = f"{output_path}\\{file_name}.tif"
    data = rasterio.open(path).read(1)
    cmap = plt.cm.get_cmap(color, 10)
    # Display the data using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap=cmap)
    plt.title(file_name)
    plt.colorbar()
    plt.show()

#FUNCTION TO DISPLAY SHAPEFILE USING FOLIUM
def display_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    gdf.to_crs(epsg=4326, inplace=True)
    geojson_data = gdf.to_json()
    mp = folium.Map(location=[gdf.centroid.y.mean(), gdf.centroid.x.mean()], zoom_start=12, 
                    control_scale=True, tiles='CartoDB Positron', name = 'CartoDB Positron')
    folium.features.GeoJson(geojson_data).add_to(mp)
    # return mp
    display(mp)


###STUDY AREA PROJECTION CHECK
desc = arcpy.Describe(aoi_path)
spatial_reference = desc.spatialReference.factoryCode
if spatial_reference != 4326:
    arcpy.management.Project(
        in_dataset=f"{aoi_path}",
        out_dataset=f"{geodatabase_path}\\study_area",
        out_coor_system='GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    )

    aoi_path = f"{geodatabase_path}\\study_area"
    print("Study area has been projected to WGS 1984 coordinate system")



###########################################
######GETTING DATA FROM GOOGLE EARTH ENGINE

aoi = geemap.shp_to_ee(aoi_path)

###NDVI DATA
try:
    print("Calculating NDVI from Landsat 9 for your study area")
    landsat = ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA') \
        .filterDate(start_date, end_date) \
        .filterBounds(aoi)

    # Function to calculate NDVI
    def calculateNDVI(image):
        return image.normalizedDifference(['B5', 'B4']).rename('NDVI')

    # Map the function over the collection
    landsat_ndvi = landsat.map(calculateNDVI)

    # Median composite of NDVI within the AOI
    median_ndvi = landsat_ndvi.median().clip(aoi)

    # Visualization parameters for NDVI
    vis_params = {
        'min': -1,
        'max': 1,
        'palette': ['blue', 'white', 'green']
    }

    # Display the NDVI result on the map
    print("Displaying NDVI result")
    Map = geemap.Map()
    Map.centerObject(aoi, 11)
    Map.addLayer(aoi, {'color': 'FF0000'}, 'Study Area')
    Map.addLayer(median_ndvi, vis_params, 'Landsat 9 NDVI')
    Map.addLayerControl()
    display(Map)
    time.sleep(5)

    # Define export parameters
    export_params = {
        'image': median_ndvi.toFloat(),
        'description': 'ndvi',
        'scale': 30,
        'region': aoi.geometry().bounds(),
        'fileFormat': 'GeoTIFF',
        'folder': f"{output_dir}",
    }

    # Export the NDVI result to Google Drive
    try:
        print("Exporting NDVI result to your Google Drive")
        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()
        vegetation_path = f"{google_drive_path}\\ndvi.tif"
    except:
        print("Failed to export NDVI")
except:
    print("Failed to calculate NDVI")



###BARE SOIL
try:
    print("Extracting bare surfaces from the study of Demattê, José AM, et al. (2020).")
    bare_surface = ee.Image('users/geocis/BareSurfaces/BS_1980_2019').clip(aoi)

    Map1 = geemap.Map()
    Map1.centerObject(aoi, 11)

    Map1.addLayer(bare_surface.select(['red', 'green', 'blue']), 
                {'min': 500, 'max': 3500, 'gamma': 1.25},
                'Bare Surface')
    
    print("Displaying Bare Surfaces")
    display(Map1)
    time.sleep(5)

    export_params_bare_soil = {
        'image': bare_surface.toFloat(),
        'description': 'bare_soil',
        'scale': 30,
        'region': aoi.geometry().bounds(),
        'fileFormat': 'GeoTIFF',
        'folder': f"{output_dir}",
    }

    try:
        print("Exporting Bare Surfaces to your Google Drive")
        task = ee.batch.Export.image.toDrive(**export_params_bare_soil)
        task.start()
        bare_soil_path = f"{google_drive_path}\\bare_soil.tif"
    except:
        print("Failed to export Bare Surfaces")
except:
    print("Bare Surface extraction process failed")


###BUILDINGS
try:
    print("Extracting Buildings from Google's Open Buildings")
    bu = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons').filterBounds(aoi)


    print("Categorizing Buildings based on the Confidence Level")
    # Filter the feature collection based on confidence levels
    t_065_070 = bu.filter(ee.Filter.rangeContains('confidence', 0.65, 0.7))
    t_070_075 = bu.filter(ee.Filter.rangeContains('confidence', 0.7, 0.75))
    t_gte_075 = bu.filter(ee.Filter.gte('confidence', 0.75))

    vis_params_065_070 = {'color': 'FF0000'}
    vis_params_070_075 = {'color': 'FFFF00'}
    vis_params_gte_075 = {'color': '00FF00'}

    Map2 = geemap.Map()
    Map2.centerObject(aoi, 11)
    Map2.addLayer(t_065_070, vis_params_065_070, 'Buildings confidence [0.65; 0.7)')
    Map2.addLayer(t_070_075, vis_params_070_075, 'Buildings confidence [0.7; 0.75)')
    Map2.addLayer(t_gte_075, vis_params_gte_075, 'Buildings confidence >= 0.75')

    print("Displaying Buildings. It may take a while to load...")
    display(Map2)
    time.sleep(300)

    print("Converting Buildings to raster data.")
    raster = bu.reduceToImage(properties=['confidence'], reducer=ee.Reducer.first())

    vis_params = {'palette': ['red']}
    Map2R = geemap.Map()
    Map2R.centerObject(aoi, 11)
    Map2R.addLayer(raster, vis_params, 'Raster')

    print("Displaying Rasterized Buildings. It may take a while to load...")
    display(Map2R)
    

    export_params_buildings = {
        'image': raster.toFloat(),
        'description': 'buildings',
        'scale': 25,
        'region': aoi.geometry().bounds(),
        'fileFormat': 'GeoTIFF',
        'folder': f"{output_dir}",
    }

    try:
        print("Exporting Buildings to Google Drive")
        task = ee.batch.Export.image.toDrive(**export_params_buildings)
        task.start()

        buildings_path = f"{google_drive_path}\\buildings.tif"
    except:
        print("Failed to export Buildings")

except:
    print("Buildings extraction process failed")

time.sleep(300)


###DEM
try:
    print("Extracting SRTM DEMS")
    srtm = ee.Image('USGS/SRTMGL1_003').clip(aoi)

    vis_params = {
        'min': -100,
        'max': 100, 
    }

    Map3 = geemap.Map()
    Map3.centerObject(aoi, 11)
    Map3.addLayer(srtm, vis_params, 'SRTM DEM')
    print("Displaying DEM")
    display(Map3)

    export_params_dem = {
        'image': srtm.toFloat(),
        'description': 'dem',
        'scale': 30, 
        'region': aoi.geometry().bounds(),
        'fileFormat': 'GeoTIFF',
        'folder': f"{output_dir}",
    }

    try:
        print("Exporting DEM to Google Drive")
        task = ee.batch.Export.image.toDrive(**export_params_dem)
        task.start()

        dem_path = f"{google_drive_path}\\dem.tif"
    except:
        print("Failed to export DEM")
    
except:
    print("DEM extraction process failed")

#WAITING TO SYNC THE DATA TO LOCAL SYSTEM
print(f"Waiting for {waiting_time} seconds to sync the files into local system")
time.sleep(waiting_time)



#CHEKING IF THE FILES HAVE BEEN SYNCED
print("Checking the availability of the files...")
def file_sync_check():
    global status
    status = 'proceed'
    if not os.path.exists(vegetation_path):
        print("No file found at the defined vegetation path")
        status = 'wait'
    if not os.path.exists(bare_soil_path):
        print("No file found at the defined bare soil path")
        status = 'wait'
    if not os.path.exists(buildings_path):
        print("No file found at the defined buildings path")
        status = 'wait'
    if not os.path.exists(dem_path):
        print("No file found at the defined DEM path")
        status = 'wait'

while True:
    print(f"{waiting_time} seconds have passed. Checking files...")
    file_sync_check()
    if status == 'wait':
        print("Files have not been synced yet. Waiting for another 30 seconds...")
        time.sleep(30)
    else:
        print("All files are available. Proceeding with the rest of the analysis...")
        break 


###DEM TO SLOPE
#DEM EXTRACTION
try:
    out_raster = arcpy.sa.ExtractByMask(dem_path, aoi_path, "INSIDE"); 
    out_raster.save(f"{geodatabase_path}\\dem_aoi")
    dem_aoi_path = f"{geodatabase_path}\\dem_aoi"
    print("DEM extraction completed")

except:
    print("Failed to extract DEM")



#SLOPE ANALYSIS
try:
    out_raster = arcpy.sa.Slope(dem_aoi_path, "DEGREE", 1, "PLANAR", "METER"); 
    out_raster.save(f"{geodatabase_path}\\slope_aoi")
    slope_aoi_path = f"{geodatabase_path}\\slope_aoi"
    print("Slope analysis completed")

    save_tif(slope_aoi_path, "slope_aoi")
    print("Slope analysis file has been saved")
    print("Displaying raster image of Slope analysis")
    try:
        display_raster("slope_aoi","rainbow")

    except:
        print("Failed to display slope analysis")
except:
    print("Slope analysis failed")

#SLOPE RESAMPLING
try:
    arcpy.management.Resample(
        in_raster=f"{slope_aoi_path}",
        out_raster=f"{geodatabase_path}\\slope_resample",
        cell_size="0.00025 0.00025",
        resampling_type="BILINEAR"
    )
    print("Slopes resampled")
    slope_resample = f"{geodatabase_path}\\slope_resample"
except:
    print("Failed to resample Streams")

#SLOPE SLICE
try:
    with arcpy.EnvManager(scratchWorkspace=f"{geodatabase_path}"):
        out_raster = arcpy.sa.Slice(
            in_raster=f"{slope_resample}",
            number_zones=5,
            slice_type="NATURAL_BREAKS",
            base_output_zone=1,
            nodata_to_value=None,
            class_interval_size=None
        )
        out_raster.save(f"{output_path}\\slope_re.tif")
        slope_re = f"{output_path}\\slope_re.tif"
        print("Slope has been reclassfied")

        try:
            print("Displaying raster image of reclassified Slope")
            display_raster("slope_re","turbo")
        except:
            print("Failed to display reclassified slope")

except:
    print("Failed to reclassify slope analysis")



###DEM TO STREAMS

#DEM FILL(DEPRESSION LESS DEM)
try:
    out_surface_raster = arcpy.sa.Fill(dem_aoi_path, None); 
    out_surface_raster.save(f"{geodatabase_path}\\dem_aoi_fill")
    dem_aoi_fill_path = f"{geodatabase_path}\\dem_aoi_fill"
    print("Depression less DEM has been created")
except:
    print("DEM fill process failed")

#FLOW DIRECTION
try:
    out_flow_direction_raster = arcpy.sa.FlowDirection(dem_aoi_fill_path, "NORMAL", None, "D8"); 
    out_flow_direction_raster.save(f"{geodatabase_path}\\flow_direction")
    flow_direction_path = f"{geodatabase_path}\\flow_direction"
    print("Flow direction has been calculated")
except:
    print("Flow direction calculation failed")

#FLOW ACCUMULATION
try:
    print("Calculating flow accumulation. It may take a while...")
    out_accumulation_raster = arcpy.sa.FlowAccumulation(flow_direction_path, None, "FLOAT", "D8"); 
    out_accumulation_raster.save(f"{geodatabase_path}\\flow_accumulation")
    flow_accumulation_path = f"{geodatabase_path}\\flow_accumulation"
    print("Flow accumulation has been calculated")
except:
    print("Failed to calculated flow accumulation")

#STREAMS DELINEATION
try:
    max = arcpy.GetRasterProperties_management(flow_accumulation_path, "MAXIMUM")
    max = float(max.getOutput(0))
    threshold = max * .1

    range = f"0 {threshold} NODATA;{threshold} {max} 1"

    out_raster = arcpy.sa.Reclassify(flow_accumulation_path, "VALUE", range, "NODATA"); 
    out_raster.save(f"{geodatabase_path}\\streams")
    streams_path = f"{geodatabase_path}\\streams"
    print("Streams have been extracted")

    save_tif(streams_path, "streams")
    print("Stream file has been saved")
    print("Displaying raster image of Streams")
    try:
        display_raster("streams","turbo")
    except:
        print("Failed to display streams")
except:
    print("Stream extraction failed")

#STREAMS EUCLIDEAN DISTANCE
try:
    with arcpy.EnvManager(mask=aoi_path):
        out_distance_raster = arcpy.sa.EucDistance(streams_path, None, 0.000277777777777786, None, "GEODESIC", None, None); 
        out_distance_raster.save(f"{geodatabase_path}\\streams_eu")
        streams_eu_path = f"{geodatabase_path}\\streams_eu"
        print("Euclidean distances from streams have been calculated")

        save_tif(streams_eu_path, "streams_eu")
        print("Euclidean distance file has been saved")
        print("Displaying raster image of Euclidean distance")
        try:
            display_raster("streams_eu","turbo")
        except:
            print("Failed to display euclidean distance raster")
except:
    print("Euclidean distance calculation failed")

#STREMS RESAMPLING
try:
    arcpy.management.Resample(
        in_raster=f"{streams_eu_path}",
        out_raster=f"{geodatabase_path}\\streams_resample",
        cell_size="0.00025 0.00025",
        resampling_type="BILINEAR"
    )
    streams_resample = f"{geodatabase_path}\\streams_resample"

    print("Streams have been resampled")
except:
    print("Failed to resample Streams")

#STREAMS RECLASSIFICATION
try:
    with arcpy.EnvManager(scratchWorkspace=f"{geodatabase_path}"):
        out_raster = arcpy.sa.Slice(
            in_raster=f"{streams_resample}",
            number_zones=5,
            slice_type="EQUAL_INTERVAL",
            base_output_zone=1,
            nodata_to_value=None,
            class_interval_size=None
        )
        out_raster.save(f"{output_path}\\streams_re.tif")
        streams_re_path = f"{output_path}\\streams_re.tif"
        print("Streams have been reclassfied")

        try:
            print("Displaying raster image of reclassified Streams")
            display_raster("streams_re","turbo")
        except:
            print("Failed to display reclassified streams")

except:
    print("Failed to reclassify Streams")

###ROADS
#CHECK REFERENCE
desc = arcpy.Describe(roads_path)
spatial_reference = desc.spatialReference.factoryCode
if spatial_reference != 4326:
    arcpy.management.Project(
        in_dataset=f"{roads_path}",
        out_dataset=f"{geodatabase_path}\\roads",
        out_coor_system='GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    )

    roads_path = f"{geodatabase_path}\\roads"
    print("Roads have been projected to WGS 1984 coordinate system")

#CLIP
try:
    arcpy.analysis.Clip(
        in_features=f"{roads_path}",
        clip_features=f"{aoi_path}",
        out_feature_class=f"{geodatabase_path}\\aoi_roads",
        cluster_tolerance=None
    )

    aoi_roads_path = f"{geodatabase_path}\\aoi_roads"
    print("Roads have been clipped according to the boundary of the study area")
    
except:
    print("Failed to clip study area roads")


#MULTIPLE RING BUFFER
distance1 = 200
distance2 = 400
distance3 = 800
distance4 = 1200
distance5 = 2000
try:
    print("Performing multiple ring buffer on roads. It may take a while...")
    arcpy.analysis.MultipleRingBuffer(
        Input_Features= f"{aoi_roads_path}",
        Output_Feature_class=f"{geodatabase_path}\\roads_buffer",
        Distances = [f"{distance1}",f"{distance2}",f"{distance3}",f"{distance4}",f"{distance5}"],
        Buffer_Unit="Meters",
        Field_Name="distance",
        Dissolve_Option="ALL",
        Outside_Polygons_Only="FULL",
        Method="GEODESIC"
    )
    roads_buffer_path = f"{geodatabase_path}\\roads_buffer"
    print("Successfully performed Multiple Ring Buffer on Roads")

    print("Saving Multiple Ring Buffer (Roads) shapefile to output folder")
    arcpy.conversion.FeatureClassToShapefile(f"{roads_buffer_path}", f"{output_path}")
    roads_buffer_path1 = f"{output_path}\\roads_buffer.shp"

except:
    print("Failed to perform multiple ring buffer on roads")


print("Displaying Multiple Ring Buffer (Roads)")
display_shapefile(roads_buffer_path1)


#CLIP MULTIPLE RING BUFFER
try:
    arcpy.analysis.Clip(
        in_features=f"{roads_buffer_path}",
        clip_features=f"{aoi_path}",
        out_feature_class=f"{geodatabase_path}\\roads_buffer_clip",
        cluster_tolerance=None
    )

    roads_buffer_clip_path = f"{geodatabase_path}\\roads_buffer_clip"
    print("Multiple Ring Buffer (Roads) has been clipped")

    try:
        arcpy.conversion.FeatureClassToShapefile(f"{roads_buffer_clip_path}", f"{output_path}")
        roads_buffer_clip_path1 = f"{output_path}\\roads_buffer_clip.shp"
        print("Saving clipped Multiple Ring Buffer (Roads) shapefile to output folder")
        
    except:
        print("Failed to save clipped Multiple Ring Buffer (Roads)")
    
except:
    print("Failed to clip Multiple Ring Buffer (Roads)")

print("Displaying Clipped Multiple Rign Buffer (Roads)")
display_shapefile(roads_buffer_clip_path1)

#POLYGON TO RASTER
try:
    arcpy.conversion.PolygonToRaster(
        in_features=f"{roads_buffer_clip_path}",
        value_field="distance",
        out_rasterdataset=f"{output_path}\\roads_re.tif",
        cell_assignment="CELL_CENTER",
        priority_field="NONE",
        cellsize=0.00025,
        build_rat="BUILD"
    )
   
    print("Multiple Ring Buffer has been converted into Raster feature")
    roads_re_path = f"{output_path}\\roads_re.tif"

    print("Displaying the rasterized Multiple Ring Buffer")
    try:
        display_raster("roads_re","turbo")
    except:
        print("Failed to display rasterized Multiple Ring Buffer")
    
except:
    print("Failed to convert Multiple Ring Buffer to raster feature")

###BARE SOIL
#EXTRACT BY MASK
try:
    out_raster = arcpy.sa.ExtractByMask(bare_soil_path, aoi_path, "INSIDE"); 
    out_raster.save(f"{geodatabase_path}\\bare_soil")
    bare_soil_path = f"{geodatabase_path}\\bare_soil"
    print("Bare Soil extraction completed")

except:
    print("Failed to extract Bare Soil")

#EXCLUDE NODATA VALUES (COPY RASTER)
try:
    arcpy.management.CopyRaster(
        in_raster=f"{bare_soil_path}",
        out_rasterdataset=f"{output_path}\\bare_soil.tif",
        config_keyword="",
        background_value=None,
        nodata_value="0",
        onebit_to_eightbit="NONE",
        colormap_to_RGB="NONE",
        pixel_type="32_BIT_FLOAT",
        scale_pixel_value="NONE",
        RGB_to_Colormap="NONE",
        format="TIFF",
        transform="NONE",
        process_as_multidimensional="CURRENT_SLICE",
        build_multidimensional_transpose="NO_TRANSPOSE"
    )

    print("NODATA values from Bare Soil have been excluded")
    bare_soil_path = f"{output_path}\\bare_soil.tif"

    print("Displaying Bare Soil raster image")
    try:
        display_raster("bare_soil","turbo")
    except:
        print("Failed to display Bare Soil raster image")

except:
    print("Failed to exclude NODATA values from Bare Soil")

#BARE SOIL RESAMPLING
try:
    arcpy.management.Resample(
        in_raster=f"{bare_soil_path}",
        out_raster=f"{geodatabase_path}\\bare_soil_resample",
        cell_size="0.00025 0.00025",
        resampling_type="BILINEAR"
    )
    print("Bare soil resampled")
    bare_soil_resample = f"{geodatabase_path}\\bare_soil_resample"
    
except:
    print("Failed to resample Bare Soil")

#RECLASSIFICATION
try:
    with arcpy.EnvManager(scratchWorkspace=f"{geodatabase_path}"):
        out_raster = arcpy.sa.Slice(
            in_raster=f"{bare_soil_resample}",
            number_zones=5,
            slice_type="NATURAL_BREAKS",
            base_output_zone=1,
            nodata_to_value=None,
            class_interval_size=None
        )
        out_raster.save(f"{output_path}\\bare_soil_re.tif")
        bare_soil_re_path = f"{output_path}\\bare_soil_re.tif"
        print("Bare Soil has been reclassified")

        try:
            print("Displaying raster image of reclassified Bare Soil")
            display_raster("bare_soil_re","turbo")
        except:
            print("Failed to display reclassified Bare Soil")

except:
    print("Bare Soil reclssification failed")



###VEGETATION 
#EXTRACT BY MASK
try:
    out_raster = arcpy.sa.ExtractByMask(vegetation_path, aoi_path, "INSIDE"); 
    out_raster.save(f"{geodatabase_path}\\vegetation")
    vegetation_path = f"{geodatabase_path}\\vegetation"
    print("Vegetation extraction completed")

except:
    print("Failed to extract Vegetation cover")

#RASTER CALCULATOR
try:
    with arcpy.EnvManager(scratchWorkspace=f"{geodatabase_path}"):
        input_raster = f"{vegetation_path}"
        output_raster = "positive_ndvi"
        expression = "SetNull(\"{0}\" <= 0, \"{0}\")".format(input_raster)
        arcpy.gp.RasterCalculator_sa(expression, output_raster)

        print("Successfully extracted positive values of NDVI")
        vegetation_path = f"{geodatabase_path}\\positive_ndvi"

except:
    print("Failed to perform raster calculator operations on NDVI")

#RESAMPLING VEGETATION
try:
    arcpy.management.Resample(
        in_raster=f"{vegetation_path}",
        out_raster=f"{geodatabase_path}\\vegetation_resample",
        cell_size="0.00025 0.00025",
        resampling_type="BILINEAR"
    )
    print("NDVI resampled")
    vegetation_resample = f"{geodatabase_path}\\vegetation_resample"
    
except:
    print("Failed to resample Bare Soil")


#RECLASSIFICATION
try:
    with arcpy.EnvManager(scratchWorkspace=f"{geodatabase_path}"):
        out_raster = arcpy.sa.Slice(
            in_raster=f"{vegetation_resample}",
            number_zones=5,
            slice_type="NATURAL_BREAKS",
            base_output_zone=1,
            nodata_to_value=None,
            class_interval_size=None
        )
    out_raster.save(f"{output_path}\\ndvi_re.tif")
    ndvi_re = f"{output_path}\\ndvi_re.tif"
    print("Successfully reclassfied NDVI")

    print("Displaying reclassified NDVI")
    try:
        display_raster("ndvi_re","turbo")
    except:
        print("Failed to display reclassified NDVI")
except:
    print("Failed to reclassify NDVI")

###BUILT UP AREAS
#EXTRACT BY MASK
try:
    out_raster = arcpy.sa.ExtractByMask(buildings_path, aoi_path, "INSIDE"); 
    out_raster.save(f"{geodatabase_path}\\buildings")
    buildings_path = f"{geodatabase_path}\\buildings"
    print("Building extraction completed")

except:
    print("Failed to extract Buildings")

#COPY RASTER TO EXCLUDE NODATA VALUES
try:
    arcpy.management.CopyRaster(
        in_raster=f"{buildings_path}",
        out_rasterdataset=f"{output_path}\\buildings.tif",
        config_keyword="",
        background_value=None,
        nodata_value="0",
        onebit_to_eightbit="NONE",
        colormap_to_RGB="NONE",
        pixel_type="32_BIT_FLOAT",
        scale_pixel_value="NONE",
        RGB_to_Colormap="NONE",
        format="TIFF",
        transform="NONE",
        process_as_multidimensional="CURRENT_SLICE",
        build_multidimensional_transpose="NO_TRANSPOSE"
    )

    print("NODATA values from Buildings have been excluded")
    buildings_path = f"{output_path}\\buildings.tif"

    print("Displaying raster image of Buildings")
    try:
        display_raster("buildings","turbo")
    except:
        print("Failed to display raster image of Buildings")

except:
    print("Failed to exclude NODATA values from Buildings")

#RESAMPLE
try:
    arcpy.management.Resample(
        in_raster=f"{buildings_path}",
        out_raster=f"{geodatabase_path}\\buildings_resample",
        cell_size="0.00025 0.00025",
        resampling_type="BILINEAR"
    )
    print("Buildings resampled")
    buildings_resample = f"{geodatabase_path}\\buildings_resample"
    
except:
    print("Failed to resample Buildings")

#EUCLIDEAN DISTANCE
try:
    with arcpy.EnvManager(mask=f"{aoi_path}", scratchWorkspace=f"{geodatabase_path}"):
        out_distance_raster = arcpy.sa.EucDistance(
            in_source_data=f"{buildings_resample}",
            maximum_distance=None,
            cell_size=0.00025,
            out_direction_raster=None,
            distance_method="PLANAR",
            in_barrier_data=None,
            out_back_direction_raster=None
        )
        out_distance_raster.save(f"{output_path}\\buildings_eu.tif")
        print("Euclidean Distance has been performed on Buildings")

        buildings_eu = f"{output_path}\\buildings_eu.tif"
except:
    print("Failed to perform Euclidean Distance on Buildings")


#RECLASSIFICATION
try:
    with arcpy.EnvManager(scratchWorkspace=f"{geodatabase_path}"):
        out_raster = arcpy.sa.Slice(
            in_raster=f"{buildings_eu}",
            number_zones=5,
            slice_type="NATURAL_BREAKS",
            base_output_zone=1,
            nodata_to_value=None,
            class_interval_size=None
        )
    out_raster.save(f"{output_path}\\buildings_re.tif")
    buildings_re = f"{output_path}\\buildings_re.tif"
    print("Successfully reclassfied Buildings")

    print("Displaying reclassified Buildings")
    try:
        display_raster("buildings_re","turbo")
    except:
        print("Failed to display reclassified Buildings")
except:
    print("Failed to reclassify Buildings")


###WEIGHTED OVERLAY

desc1 = f"{roads_re_path} {road_weight} 'Value' ({distance1} {road_matrix[0]}; {distance2} {road_matrix[1]}; {distance3} {road_matrix[2]}; {distance4} {road_matrix[3]}; {distance5} {road_matrix[4]}; NODATA NODATA)"
desc2 = f"{streams_re_path} {stream_weight} 'Value' (1 {stream_matrix[0]}; 2 {stream_matrix[1]}; 3 {stream_matrix[2]}; 4 {stream_matrix[3]}; 5 {stream_matrix[4]}; NODATA NODATA)"
desc3 = f"{slope_re} {slope_weight} 'Value' (1 {slope_matrix[0]}; 2 {slope_matrix[1]}; 3 {slope_matrix[2]}; 4 {slope_matrix[3]}; 5 {slope_matrix[4]}; NODATA NODATA)"
desc4 = f"{ndvi_re} {vegetation_weight} 'Value' (1 {vegetation_matrix[0]}; 2 {vegetation_matrix[1]}; 3 {vegetation_matrix[2]}; 4 {vegetation_matrix[3]}; 5 {vegetation_matrix[4]}; NODATA NODATA)"
desc5 = f"{buildings_re} {buildings_weight} 'Value' (1 {buildings_matrix[0]}; 2 {buildings_matrix[1]}; 3 {buildings_matrix[2]}; 4 {buildings_matrix[3]}; 5 {buildings_matrix[4]}; NODATA NODATA)"
desc6 = f"{bare_soil_re_path} {bare_soil_weight} 'Value' (1 {bare_soil_matrix[0]}; 2 {bare_soil_matrix[1]}; 3 {bare_soil_matrix[2]}; 4 {bare_soil_matrix[3]}; 5 {bare_soil_matrix[4]}; NODATA NODATA)"

try:
    with arcpy.EnvManager(cellSize="MINOF", scratchWorkspace=f"{geodatabase_path}"):
        out_raster = arcpy.sa.WeightedOverlay(
            in_weighted_overlay_table=f"({desc1}; {desc2}; {desc3}; {desc4}; {desc5}; {desc6});1 5 1"
        )
        out_raster.save(f"{output_path}\\landfill_suitable_sites.tif")
        print("Successfully performed Landfill suitability analysis")

except:
    print("Failed to perform weighted overlay")


print("Process Completed!")
