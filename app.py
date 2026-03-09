import os
import cv2
import rasterio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns  # Added for high-quality charts
from flask import Flask, render_template, request, send_file, jsonify, session, url_for
from ultralytics import YOLO
from math import pi
import math
import base64
from io import BytesIO
import glob
import shutil
import atexit
import uuid
from osgeo import gdal
from geographiclib.geodesic import Geodesic

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Needed for session management

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurable settings
PATCH_SIZE = 500
YOLO_SIZE = 512
PIXEL_SCALE = 100.0
CRATER_CONF_THRESHOLD = 0.7
CRATER_NMS_THRESHOLD = 0.7
SHADOW_CONF_THRESHOLD = 0.1
CANNY_LOW = 20
CANNY_HIGH = 80
MIN_CONTOUR_AREA = 150
RADIUS_SCALE_FACTOR = 1.1

# Moon radius for solar geometry calculation
MOON_RADIUS = 1737400  # meters

# Load YOLO models
CRATER_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_final2.pt')
SHADOW_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best200shadow.pt')
try:
    crater_model = YOLO(CRATER_MODEL_PATH)
    shadow_model = YOLO(SHADOW_MODEL_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Helper function to encode image to base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# Helper function to save base64 image to file (FIXED)
def save_base64_image(base64_str, filename):
    with open(os.path.join(OUTPUT_DIR, filename), 'wb') as f:
        f.write(base64.b64decode(base64_str))  # Correctly decode the base64_str

# TIFF Extraction Function
def extract_band(input_file, output_file, band_number, driver_name, scale=False):
    ds = gdal.Open(input_file)
    if ds is None:
        print(f"Error: Could not open {input_file}. Please check if the file is a valid GeoTIFF.")
        return False
    if band_number > ds.RasterCount:
        print(f"Error: Band {band_number} not found in {input_file} (only {ds.RasterCount} bands available)")
        return False

    if scale and driver_name == "PNG":
        band = ds.GetRasterBand(band_number)
        band_data = band.ReadAsArray()
        min_val, max_val = np.percentile(band_data, [2, 98])
        gdal.Translate(output_file, ds, bandList=[band_number], scaleParams=[[min_val, max_val, 0, 255]],
                       format=driver_name)
    else:
        gdal.Translate(output_file, ds, bandList=[band_number], format=driver_name)

    print(f"Saved {output_file}")
    return True

# Solar Geometry Calculation Functions
def normalize_lon(lon):
    return ((lon + 180) % 360) - 180

def calculate_solar_geometry(observer_lat, observer_lon, subsolar_lat, subsolar_lon):
    observer_lon = normalize_lon(observer_lon)
    subsolar_lon = normalize_lon(subsolar_lon)

    geo = Geodesic(MOON_RADIUS, 0)
    g = geo.Inverse(observer_lat, observer_lon, subsolar_lat, subsolar_lon)
    surface_distance_m = g['s12']
    azimuth_deg = g['azi1'] % 360

    central_angle_rad = surface_distance_m / MOON_RADIUS
    solar_elevation_deg = 90 - math.degrees(central_angle_rad)

    return round(solar_elevation_deg, 2), round(azimuth_deg, 2)

# Detect craters using YOLO (shared between shadow and DEM methods)
def detect_craters(image):
    pad_width = (YOLO_SIZE - PATCH_SIZE) // 2
    image_padded = cv2.copyMakeBorder(image, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
    image_bgr = cv2.cvtColor(image_padded, cv2.COLOR_GRAY2BGR)

    results = crater_model.predict(image_bgr, conf=CRATER_CONF_THRESHOLD, iou=CRATER_NMS_THRESHOLD, classes=[0], imgsz=YOLO_SIZE)

    craters = []
    for result in results:
        if not result.boxes:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confidences):
            x_min, y_min, x_max, y_max = map(int, box)
            x_min, y_min = x_min - pad_width, y_min - pad_width
            x_max, y_max = x_max - pad_width, y_max - pad_width

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(PATCH_SIZE, x_max), min(PATCH_SIZE, y_max)
            if x_max <= x_min or y_max <= y_min:
                continue

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            radius = max((x_max - x_min) // 2, (y_max - y_min) // 2)
            radius = min(radius, PATCH_SIZE - max(center_x, center_y))
            radius *= RADIUS_SCALE_FACTOR

            craters.append({
                'center_x': center_x,
                'center_y': center_y,
                'radius': radius,
                'confidence': conf
            })

    return craters

# Shadow-based Detection Method
def shadow_detection(png_path, sun_elevation_deg, sun_azimuth_deg, craters, paired_ids=None):
    sun_elevation_rad = math.radians(sun_elevation_deg)
    opposite_azimuth_deg = (sun_azimuth_deg + 180) % 360
    theta = math.radians(90 - opposite_azimuth_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at {png_path}")
    if image.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Expected {PATCH_SIZE}x{PATCH_SIZE}, got {image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    pad_width = (YOLO_SIZE - PATCH_SIZE) // 2
    image_padded = cv2.copyMakeBorder(image, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
    image_bgr = cv2.cvtColor(image_padded, cv2.COLOR_GRAY2BGR)

    base_name = os.path.splitext(os.path.basename(png_path))[0]
    shadow_results = shadow_model.predict(image_bgr, conf=SHADOW_CONF_THRESHOLD, imgsz=YOLO_SIZE)

    shadow_masks = []
    shadow_confidences = []
    for result in shadow_results:
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (YOLO_SIZE, YOLO_SIZE), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8) * 255
                mask = mask[pad_width:pad_width+PATCH_SIZE, pad_width:pad_width+PATCH_SIZE]
                shadow_masks.append(mask)
                conf = result.boxes.conf[i].item() if result.boxes is not None and i < len(result.boxes.conf) else 0.0
                shadow_confidences.append(conf)

    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detection_data = []
    for crater_idx, crater in enumerate(craters):
        center_x = int(crater['center_x'])
        center_y = int(crater['center_y'])
        radius = crater['radius']
        confidence = crater['confidence']

        crater_id = f"Crater at ({center_x}, {center_y})"
        if paired_ids and crater_id not in paired_ids:
            continue

        shadow_mask = None
        shadow_conf = 0.0
        x1, y1 = max(0, int(center_x - radius)), max(0, int(center_y - radius))
        x2, y2 = min(PATCH_SIZE, int(center_x + radius)), min(PATCH_SIZE, int(center_y + radius))
        for idx, mask in enumerate(shadow_masks):
            mask_roi = mask[y1:y2, x1:x2]
            if np.any(mask_roi == 255):
                shadow_mask = mask
                shadow_conf = shadow_confidences[idx]
                break

        patch_number = base_name

        cv2.circle(output_image, (center_x, center_y), int(radius), (0, 255, 0), 2)
        diameter_m = 2 * radius * PIXEL_SCALE

        depth_m = 0.0
        shadow_length_m = 0.0
        if shadow_mask is not None:
            edge_x = center_x - radius * dx
            edge_y = center_y - radius * dy
            edge_point = (int(edge_x), int(edge_y))

            max_dist = 0
            end_point = None
            max_shadow_length = radius * 2
            search_range = int(max_shadow_length)
            in_shadow = False
            for t in np.linspace(0, search_range, 400):
                px = int(edge_x + t * dx)
                py = int(edge_y + t * dy)
                if 0 <= py < PATCH_SIZE and 0 <= px < PATCH_SIZE:
                    if shadow_mask[py, px] == 255:
                        in_shadow = True
                        dist = math.sqrt((px - edge_x)**2 + (py - edge_y)**2)
                        if dist > max_dist:
                            max_dist = dist
                            end_point = (px, py)
                    elif in_shadow and shadow_mask[py, px] == 0:
                        break

            if end_point:
                cv2.line(output_image, edge_point, end_point, (0, 255, 0), 2)
                shadow_length_pixels = math.sqrt((end_point[0] - edge_point[0])**2 + (end_point[1] - edge_point[1])**2)
                shadow_length_m = shadow_length_pixels * PIXEL_SCALE
                depth_m = shadow_length_m * math.tan(sun_elevation_rad)

        radius_m = radius * PIXEL_SCALE
        volume_m3 = (math.pi / 6) * depth_m * (3 * radius_m**2 + depth_m**2) if depth_m > 0 else 0.0

        detection_data.append({
            "patch_number": patch_number,
            "crater_id": crater_id,
            "row": 0,
            "col": 0,
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius,
            "diameter_m": diameter_m,
            "confidence": confidence,
            "shadow_length_m": shadow_length_m,
            "depth_m": depth_m,
            "volume_m3": volume_m3
        })

    detection_output_path = os.path.join(OUTPUT_DIR, f"shadow_detection_{base_name}.png")
    cv2.imwrite(detection_output_path, output_image)

    # Combine shadow results for display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")
    ax2.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Shadow Detection")
    ax2.axis("off")
    plt.tight_layout()
    shadow_combined_path = os.path.join(OUTPUT_DIR, "shadow_combined.png")
    plt.savefig(shadow_combined_path, dpi=600, bbox_inches="tight")
    plt.close()

    return detection_data, detection_output_path, image

# DEM-based Detection Method
def dem_detection(png_path, tiff_path, craters, paired_ids=None):
    image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at {png_path}")
    if image.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Expected {PATCH_SIZE}x{PATCH_SIZE}, got {image.shape}")

    with rasterio.open(tiff_path) as src:
        elev_data = src.read(1)
        if elev_data.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"TIFF dimensions invalid: {elev_data.shape}")

        img_detected = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        base_name = os.path.splitext(os.path.basename(png_path))[0]
        detection_data = []

        for idx, crater in enumerate(craters, 1):
            center_x = crater['center_x']
            center_y = crater['center_y']
            radius = crater['radius']
            conf = crater['confidence']

            crater_id = f"Crater at ({center_x}, {center_y})"
            if paired_ids and crater_id not in paired_ids:
                continue

            lon, lat = src.xy(center_y, center_x)

            cv2.circle(img_detected, (center_x, center_y), int(radius), (0, 255, 0), 3)
            diameter_m = 2 * radius * PIXEL_SCALE

            mask = np.zeros_like(elev_data, dtype=bool)
            y, x = np.ogrid[:PATCH_SIZE, :PATCH_SIZE]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            crater_elev = elev_data[mask]
            valid_elev = crater_elev[~np.isnan(crater_elev)]
            lowest_elev = valid_elev.min() if valid_elev.size > 0 else np.nan

            ring_mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= (radius + 5) ** 2) & \
                        ((x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2)
            rim_elev = elev_data[ring_mask]
            valid_rim = rim_elev[~np.isnan(rim_elev)]
            rim_elev_mean = valid_rim.mean() if valid_rim.size > 0 else lowest_elev

            depth_m = (rim_elev_mean - lowest_elev) * 1000 if valid_elev.size > 0 and not np.isnan(lowest_elev) else 0.0
            radius_m = radius * PIXEL_SCALE
            volume_m3 = (pi / 6) * depth_m * (3 * radius_m ** 2 + depth_m ** 2) if depth_m > 0 else 0.0

            detection_data.append({
                "patch_number": base_name,
                "crater_id": crater_id,
                "row": 0,
                "col": 0,
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius,
                "diameter_m": diameter_m,
                "confidence": conf,
                "lowest_elevation": lowest_elev,
                "depth_m": depth_m,
                "volume_m3": volume_m3
            })

        elev_map = elev_data.copy().astype(float)
        elev_map[np.isnan(elev_data)] = np.nan
        valid_elev = elev_map[~np.isnan(elev_map)]
        if valid_elev.size > 0:
            norm = plt.Normalize(vmin=valid_elev.min(), vmax=valid_elev.max())
            cmap = plt.cm.RdYlGn
            elev_map_rgb = (cmap(norm(elev_map)) * 255).astype(np.uint8)[:, :, :3]
            elev_map_rgb = cv2.cvtColor(elev_map_rgb, cv2.COLOR_RGB2BGR)
        else:
            elev_map_rgb = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(image, cmap="gray")
        ax1.set_title("Original PNG")
        ax1.axis("off")
        ax2.imshow(img_detected)
        ax2.set_title("Detected Craters (Circles)")
        ax2.axis("off")
        ax3.imshow(elev_map_rgb)
        ax3.set_title("Elevation Map (Red=Low, Green=High)")
        ax3.axis("off")
        plt.tight_layout()

        vis_path = os.path.join(OUTPUT_DIR, f"dem_visualization_{base_name}.png")
        plt.savefig(vis_path, dpi=600, bbox_inches="tight")
        plt.close()

    return detection_data, vis_path, image, img_detected, elev_map_rgb

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['GET', 'POST'])
def extract_page():
    extracted_files = session.get('extracted_files', None)
    solar_results = session.get('solar_results', None)

    # Initialize default values for form fields to avoid KeyError
    obs_lat = solar_results.get('obs_lat', '') if solar_results else ''
    obs_lon = solar_results.get('obs_lon', '') if solar_results else ''
    sub_lat = solar_results.get('sub_lat', '') if solar_results else ''
    sub_lon = solar_results.get('sub_lon', '') if solar_results else ''

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'extract':
            tiff_file = request.files.get('tiff_file')
            if not tiff_file:
                return "No TIFF file uploaded. Please upload a valid GeoTIFF file."

            input_path = os.path.join(OUTPUT_DIR, 'raw_input.tiff')
            tiff_file.save(input_path)

            base_name = str(uuid.uuid4())
            png_file = os.path.join(OUTPUT_DIR, f"{base_name}.png")
            tiff_file = os.path.join(OUTPUT_DIR, f"{base_name}.tif")

            success_png = extract_band(input_path, png_file, 1, "PNG", scale=True)
            success_tiff = extract_band(input_path, tiff_file, 6, "GTiff")

            if success_png and success_tiff:
                extracted_files = {
                    'png': f"{base_name}.png",
                    'tiff': f"{base_name}.tif",
                    'png_path': png_file,
                    'tiff_path': tiff_file
                }
                session['extracted_files'] = extracted_files
            else:
                return "Extraction failed. Please check the input TIFF file."

        elif action == 'calculate':
            try:
                obs_lat = float(request.form['obs_lat'])
                obs_lon = float(request.form['obs_lon'])
                sub_lat = float(request.form['sub_lat'])
                sub_lon = float(request.form['sub_lon'])

                elevation, azimuth = calculate_solar_geometry(obs_lat, obs_lon, sub_lat, sub_lon)
                solar_results = {
                    'obs_lat': obs_lat,
                    'obs_lon': obs_lon,
                    'sub_lat': sub_lat,
                    'sub_lon': sub_lon,
                    'elevation': elevation,
                    'azimuth': azimuth
                }
                session['solar_results'] = solar_results

                # Update form field values
                obs_lat = solar_results['obs_lat']
                obs_lon = solar_results['obs_lon']
                sub_lat = solar_results['sub_lat']
                sub_lon = solar_results['sub_lon']
            except ValueError:
                return "Invalid input for solar geometry calculation. Please enter numeric values."

    return render_template('extract.html', extracted_files=extracted_files, solar_results=solar_results,
                           obs_lat=obs_lat, obs_lon=obs_lon, sub_lat=sub_lat, sub_lon=sub_lon)

@app.route('/download_file/<filename>')
def download_file(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found."

@app.route('/download_shadow_image')
def download_shadow_image():
    shadow_path = session.get('detection_results', {}).get('shadow_path')
    if shadow_path and os.path.exists(os.path.join(OUTPUT_DIR, shadow_path)):
        return send_file(os.path.join(OUTPUT_DIR, shadow_path), as_attachment=True, download_name='shadow_detection.png')
    return "Shadow detection image not found."

@app.route('/download_dem_image')
def download_dem_image():
    dem_detected_path = session.get('detection_results', {}).get('dem_detected_path')
    if dem_detected_path and os.path.exists(os.path.join(OUTPUT_DIR, dem_detected_path)):
        return send_file(os.path.join(OUTPUT_DIR, dem_detected_path), as_attachment=True, download_name='dem_detection.png')
    return "DEM detection image not found."

@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(OUTPUT_DIR, 'combined_results.csv')
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name='combined_results.csv')
    return "CSV not found."

@app.route('/download_chart/<chart_type>')
def download_chart(chart_type):
    chart_map = {
        'diameter': 'diameter_comparison.png',
        'depth': 'depth_comparison.png',
        'volume': 'volume_comparison.png'
    }
    chart_file = chart_map.get(chart_type)
    if chart_file and os.path.exists(os.path.join(OUTPUT_DIR, chart_file)):
        return send_file(os.path.join(OUTPUT_DIR, chart_file), as_attachment=True, download_name=f'{chart_type}_comparison.png')
    return f"{chart_type} chart not found."

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    sun_elevation = session.get('solar_results', {}).get('elevation', "eg: 0.62")
    sun_azimuth = session.get('solar_results', {}).get('azimuth', "eg: 272.06")

    if request.method == 'POST':
        extracted_files = session.get('extracted_files', None)
        solar_results = session.get('solar_results', None)

        if 'png_file_path' in request.form and 'tiff_file_path' in request.form:
            png_path = request.form['png_file_path']
            tiff_path = request.form['tiff_file_path']
            try:
                sun_elevation = float(request.form['sun_elevation'])
                sun_azimuth = float(request.form['sun_azimuth'])
            except ValueError:
                return "Invalid sun elevation or azimuth values. Please ensure they are numeric."
        else:
            # Clean up previous detection files
            detection_files = [
                'shadow_combined.png', 'dem_visualization_input.png',
                'original_*.png', 'shadow_*.png', 'dem_detected_*.png', 'elev_map_*.png',
                'combined_results.csv'
            ]
            for pattern in detection_files:
                for file_path in glob.glob(os.path.join(OUTPUT_DIR, pattern)):
                    if os.path.exists(file_path):
                        os.remove(file_path)

            if 'detection_results' in session:
                del session['detection_results']

            png_file = request.files.get('png_file')
            tiff_file = request.files.get('tiff_file')
            if not png_file or not tiff_file:
                return "Please upload both PNG and TIFF files."

            try:
                sun_elevation = float(request.form['sun_elevation'])
                sun_azimuth = float(request.form['sun_azimuth'])
            except ValueError:
                return "Invalid sun elevation or azimuth values. Please ensure they are numeric."

            png_path = os.path.join(OUTPUT_DIR, 'input.png')
            tiff_path = os.path.join(OUTPUT_DIR, 'input.tif')
            png_file.save(png_path)
            tiff_file.save(tiff_path)

        if not os.path.exists(png_path):
            return f"Error: PNG file not found at {png_path}. Please check the file path."
        if not os.path.exists(tiff_path):
            return f"Error: TIFF file not found at {tiff_path}. Please check the file path."

        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return f"Error: Failed to load image at {png_path}. Please check the file."
        if image.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Expected {PATCH_SIZE}x{PATCH_SIZE}, got {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)

        craters = detect_craters(image)

        shadow_data, shadow_output_path, original_img = shadow_detection(png_path, float(sun_elevation), float(sun_azimuth), craters)
        dem_data, dem_output_path, _, dem_detected, elev_map_rgb = dem_detection(png_path, tiff_path, craters)

        shadow_ids = {d['crater_id'] for d in shadow_data}
        dem_ids = {d['crater_id'] for d in dem_data}
        paired_ids = shadow_ids.intersection(dem_ids)

        if not paired_ids:
            return "No paired craters detected. Please check your input files or adjust detection parameters."

        shadow_data, shadow_output_path, _ = shadow_detection(png_path, float(sun_elevation), float(sun_azimuth), craters, paired_ids)
        shadow_img = cv2.imread(shadow_output_path)
        dem_data, dem_output_path, _, _, _ = dem_detection(png_path, tiff_path, craters, paired_ids)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original_img, cmap="gray")
        ax1.set_title("Original Image")
        ax1.axis("off")
        ax2.imshow(cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB))
        ax2.set_title("Shadow Detection")
        ax2.axis("off")
        plt.tight_layout()
        shadow_combined_path = os.path.join(OUTPUT_DIR, "shadow_combined.png")
        plt.savefig(shadow_combined_path, dpi=600, bbox_inches="tight")
        plt.close()

        combined_data = []
        paired_craters = []
        for shadow_d, dem_d in zip(shadow_data, dem_data):
            if (shadow_d['crater_id'] in paired_ids and dem_d['crater_id'] in paired_ids and
                shadow_d['crater_id'] == dem_d['crater_id'] and shadow_d['shadow_length_m'] > 0.1):
                paired_craters.append(shadow_d['crater_id'])
                combined_data.append({
                    'crater_id': shadow_d['crater_id'],
                    'shadow': {
                        'center_x': float(shadow_d['center_x']),
                        'center_y': float(shadow_d['center_y']),
                        'diameter_m': float(shadow_d['diameter_m']),
                        'depth_m': float(shadow_d['depth_m']),
                        'volume_m3': float(shadow_d['volume_m3'])
                    },
                    'dem': {
                        'center_x': float(dem_d['center_x']),
                        'center_y': float(dem_d['center_y']),
                        'diameter_m': float(dem_d['diameter_m']),
                        'depth_m': float(dem_d['depth_m']),
                        'volume_m3': float(dem_d['volume_m3'])
                    }
                })

        df = pd.DataFrame([
            {**{'method': 'Shadow', 'crater_id': d['crater_id']}, **{k: v for k, v in d['shadow'].items()}} for d in combined_data
        ] + [
            {**{'method': 'DEM', 'crater_id': d['crater_id']}, **{k: v for k, v in d['dem'].items()}} for d in combined_data
        ])
        csv_path = os.path.join(OUTPUT_DIR, 'combined_results.csv')
        df.to_csv(csv_path, index=False)

        original_filename = f"original_{uuid.uuid4()}.png"
        shadow_filename = f"shadow_{uuid.uuid4()}.png"
        dem_detected_filename = f"dem_detected_{uuid.uuid4()}.png"
        elev_map_filename = f"elev_map_{uuid.uuid4()}.png"
        original_base64 = encode_image_to_base64(original_img)
        shadow_base64 = encode_image_to_base64(shadow_img)
        dem_detected_base64 = encode_image_to_base64(dem_detected)
        elev_map_base64 = encode_image_to_base64(elev_map_rgb)
        save_base64_image(original_base64, original_filename)
        save_base64_image(shadow_base64, shadow_filename)
        save_base64_image(dem_detected_base64, dem_detected_filename)
        save_base64_image(elev_map_base64, elev_map_filename)

        session['detection_results'] = {
            'shadow_combined': shadow_combined_path.split('/')[-1],
            'dem_combined': dem_output_path.split('/')[-1],
            'original_path': original_filename,
            'shadow_path': shadow_filename,
            'dem_detected_path': dem_detected_filename,
            'elev_map_path': elev_map_filename,
            'combined_craters': combined_data,
            'paired_craters': paired_craters
        }

        if extracted_files:
            session['extracted_files'] = extracted_files
        if solar_results:
            session['solar_results'] = solar_results

        return render_template('input.html', sun_elevation=sun_elevation, sun_azimuth=sun_azimuth, **session['detection_results'])

    if 'detection_results' in session:
        return render_template('input.html', sun_elevation=sun_elevation, sun_azimuth=sun_azimuth, **session['detection_results'])
    return render_template('input.html', sun_elevation=sun_elevation, sun_azimuth=sun_azimuth)

@app.route('/results')
def results():
    csv_path = os.path.join(OUTPUT_DIR, 'combined_results.csv')
    if not os.path.exists(csv_path):
        return "No results available. Please run the detection first."

    # Read the CSV file for crater data to display in the table
    df = pd.read_csv(csv_path)
    crater_data = df.to_dict('records')  # Convert DataFrame to a list of dictionaries for the template

    grouped = df.groupby('crater_id')
    crater_ids = list(grouped.groups.keys())
    num_craters = len(crater_ids)

    if num_craters == 0:
        return "No paired craters detected. Please run the detection first."

    shadow_df = df[df['method'] == 'Shadow']
    dem_df = df[df['method'] == 'DEM']

    shadow_mean_diameter = shadow_df['diameter_m'].mean() if not shadow_df.empty else 0
    dem_mean_diameter = dem_df['diameter_m'].mean() if not dem_df.empty else 0
    shadow_mean_depth = shadow_df['depth_m'].mean() if not shadow_df.empty else 0
    dem_mean_depth = dem_df['depth_m'].mean() if not dem_df.empty else 0
    shadow_mean_volume = shadow_df['volume_m3'].mean() if not shadow_df.empty else 0
    dem_mean_volume = dem_df['volume_m3'].mean() if not dem_df.empty else 0

    diameter_diff = (abs(shadow_mean_diameter - dem_mean_diameter) / max(shadow_mean_diameter, dem_mean_diameter, 1)) * 100
    depth_diff = (abs(shadow_mean_depth - dem_mean_depth) / max(shadow_mean_depth, dem_mean_depth, 1)) * 100
    volume_diff = (abs(shadow_mean_volume - dem_mean_volume) / max(shadow_mean_volume, dem_mean_volume, 1)) * 100

    combined_difference = (diameter_diff + depth_diff + volume_diff) / 3
    combined_accuracy = 100 - combined_difference

    chart_files = []
    if num_craters > 0:
        shadow_diameters = shadow_df['diameter_m'].tolist()
        dem_diameters = dem_df['diameter_m'].tolist()
        shadow_depths = shadow_df['depth_m'].tolist()
        dem_depths = dem_df['depth_m'].tolist()
        shadow_volumes = shadow_df['volume_m3'].tolist()
        dem_volumes = dem_df['volume_m3'].tolist()

        # Prepare positions for bars
        x = np.arange(len(crater_ids))
        width = 0.35

        # Diameter Chart
        plt.figure(figsize=(12, 5))
        sns.set_style("whitegrid")
        plt.bar(x - width/2, shadow_diameters, width, label='Shadow', color='#1f77b4', edgecolor='black')
        plt.bar(x + width/2, dem_diameters, width, label='DEM', color='#ff7f0e', edgecolor='black')
        plt.xlabel('Crater ID')
        plt.ylabel('Diameter (m)')
        plt.title('Diameter Comparison (Shadow vs DEM)')
        plt.xticks(x, crater_ids, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        diameter_chart_path = os.path.join(OUTPUT_DIR, 'diameter_comparison.png')
        plt.savefig(diameter_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append('diameter_comparison.png')

        # Depth Chart
        plt.figure(figsize=(12, 5))
        sns.set_style("whitegrid")
        plt.bar(x - width/2, shadow_depths, width, label='Shadow', color='#1f77b4', edgecolor='black')
        plt.bar(x + width/2, dem_depths, width, label='DEM', color='#ff7f0e', edgecolor='black')
        plt.xlabel('Crater ID')
        plt.ylabel('Depth (m)')
        plt.title('Depth Comparison (Shadow vs DEM)')
        plt.xticks(x, crater_ids, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        depth_chart_path = os.path.join(OUTPUT_DIR, 'depth_comparison.png')
        plt.savefig(depth_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append('depth_comparison.png')

        # Volume Chart
        plt.figure(figsize=(12, 5))
        sns.set_style("whitegrid")
        plt.bar(x - width/2, shadow_volumes, width, label='Shadow', color='#1f77b4', edgecolor='black')
        plt.bar(x + width/2, dem_volumes, width, label='DEM', color='#ff7f0e', edgecolor='black')
        plt.yscale('log')
        plt.xlabel('Crater ID')
        plt.ylabel('Volume (m³, log scale)')
        plt.title('Volume Comparison (Shadow vs DEM)')
        plt.xticks(x, crater_ids, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        volume_chart_path = os.path.join(OUTPUT_DIR, 'volume_comparison.png')
        plt.savefig(volume_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append('volume_comparison.png')

    return render_template('results.html',
                           csv_path='combined_results.csv',
                           comparison_charts=chart_files,
                           combined_accuracy=combined_accuracy,
                           combined_difference=combined_difference,
                           num_craters=num_craters,
                           crater_data=crater_data)

@app.route('/about')
def about():
    return render_template('about.html')

# Cleanup on exit
@atexit.register
def cleanup():
    print("Cleaning up resources...")
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == '__main__':
    app.run(debug=True)