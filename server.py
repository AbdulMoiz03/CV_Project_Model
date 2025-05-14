import os
import uuid
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, send_from_directory
import folium
from folium import plugins

# Constants
REFERENCE_SIZES = {
    'BusinessDepartment': {'height': 15.0, 'width': 40.0, 'reference_pixel_height': 450},
    'Library': {'height': 11.0, 'width': 45.0, 'reference_pixel_height': 450},
    'EEDepartment': {'height': 12.0, 'width': 50.0, 'reference_pixel_height': 450},
    'CivilDepartment': {'height': 14.0, 'width': 50.0, 'reference_pixel_height': 500},
    'OldCSDepartment': {'height': 15.0, 'width': 48.0, 'reference_pixel_height': 450},
    'NewCSDepartment': {'height': 16.0, 'width': 50.0, 'reference_pixel_height': 500}
}

BUILDING_COORDINATES = {
    'BusinessDepartment': (33.6423, 73.0751),  # Example coordinates
    'EEDepartment': (33.6425, 73.0753),
    'Library': (33.6422, 73.0752),
    'OldCSDepartment': (33.6424, 73.0754),
    'CivilDepartment': (33.6421, 73.0755),
    'NewCSDepartment': (33.6426, 73.0756)
}

def load_recognition_model():
    """Loads the trained landmark recognition model"""
    # Load model
    model_path = 'model/landmark_recognition_model.h5'
    if not os.path.exists(model_path):
        model_path = 'landmark_recognition_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Please ensure landmark_recognition_model.h5 is in the correct location.")

    model = load_model(model_path)

    # Load annotations
    annotations_path = "annotations/annotations.csv"
    if not os.path.exists(annotations_path):
        annotations_path = "annotations.csv"
    if not os.path.exists(annotations_path):
        raise FileNotFoundError("Annotations CSV not found. Please ensure annotations.csv is in the correct location.")

    df = pd.read_csv(annotations_path)
    label_encoder = LabelEncoder()
    label_encoder.fit(df['building_name'])

    # Initialize base model
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False

    return model, label_encoder, base_model

def detect_building_boundaries(image_path):
    """Detect the boundaries of a building in the image"""
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0, img_rgb

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return (x, y, w, h), h, img_rgb

def estimate_distance_by_size(building_name, pixel_height):
    """Estimate the distance to a building based on its apparent size in pixels"""
    if building_name not in REFERENCE_SIZES:
        return None

    reference_data = REFERENCE_SIZES[building_name]
    reference_height_m = reference_data['height']
    reference_pixel_height = reference_data['reference_pixel_height']
    reference_distance = 10.0  # meters
    focal_length = (reference_pixel_height * reference_distance) / reference_height_m
    estimated_distance = (reference_height_m * focal_length) / pixel_height

    return estimated_distance

def calculate_distance_between_points(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c * 1000  # Convert to meters

    return distance

def estimate_location(building_names, distances, bearings=None):
    """Estimate user's location based on distances to landmarks"""
    try:
        if not building_names or not distances:
            return None

        if len(building_names) >= 3:
            return estimate_location_by_trilateration(building_names[:3], distances[:3])
        elif len(building_names) == 2:
            return estimate_location_by_bilateration(building_names, distances)
        elif len(building_names) == 1:
            return estimate_location_by_single_building(building_names[0], distances[0], bearings)
        return None
    except Exception as e:
        print(f"Error in estimate_location: {str(e)}")
        return None

def estimate_location_by_single_building(building_name, distance, bearings=None):
    """Estimate user location using a single building with improved accuracy"""
    try:
        if building_name not in BUILDING_COORDINATES:
            return None

        import math
        
        building_lat, building_lon = BUILDING_COORDINATES[building_name]
        
        # Convert distance from meters to degrees (approximate)
        lat_degree = distance / 111320.0  # 1 degree = 111.32 km
        lon_degree = distance / (111320.0 * math.cos(math.radians(building_lat)))
        
        if bearings is not None and len(bearings) > 0:
            # Use provided bearing
            bearing_rad = math.radians(bearings[0])
            
            # Calculate new position
            new_lat = building_lat + (lat_degree * math.cos(bearing_rad))
            new_lon = building_lon + (lon_degree * math.sin(bearing_rad))
            
            return (new_lat, new_lon)
        else:
            # Try multiple angles and find the most likely position
            possible_positions = []
            for angle in range(0, 360, 45):  # Try 8 different directions
                bearing_rad = math.radians(angle)
                new_lat = building_lat + (lat_degree * math.cos(bearing_rad))
                new_lon = building_lon + (lon_degree * math.sin(bearing_rad))
                
                # Check if this position is valid
                if is_valid_location((new_lat, new_lon), [building_name]):
                    possible_positions.append((new_lat, new_lon))
            
            if possible_positions:
                # Return the position that's most likely based on campus layout
                return find_best_position(possible_positions, building_name)
            
            # Fallback to south direction
            return (building_lat - lat_degree, building_lon)
            
    except Exception as e:
        print(f"Error in single building estimation: {str(e)}")
        return None

def estimate_location_by_bilateration(buildings, distances):
    """Estimate user location using two buildings"""
    try:
        if len(buildings) != 2 or len(distances) != 2:
            return None

        import math
        import numpy as np
        
        # Get coordinates of both buildings
        lat1, lon1 = BUILDING_COORDINATES[buildings[0]]
        lat2, lon2 = BUILDING_COORDINATES[buildings[1]]
        r1, r2 = distances
        
        # Convert to meters
        actual_distance = calculate_distance_between_points(lat1, lon1, lat2, lon2)
        
        # Check if the distances are valid
        if actual_distance > r1 + r2 or actual_distance < abs(r1 - r2):
            return None
            
        # Convert distances to degrees for calculation
        r1_deg = r1 / 111320.0
        r2_deg = r2 / 111320.0
        
        # Calculate intersection points
        try:
            # Use law of cosines to find angles
            cos_a = (r1_deg**2 + actual_distance**2 - r2_deg**2) / (2 * r1_deg * actual_distance)
            if abs(cos_a) > 1:
                return None
                
            angle_a = math.acos(cos_a)
            bearing = math.atan2(lon2 - lon1, lat2 - lat1)
            
            # Calculate two possible points
            lat3_1 = lat1 + r1_deg * math.cos(bearing + angle_a)
            lon3_1 = lon1 + r1_deg * math.sin(bearing + angle_a)
            
            lat3_2 = lat1 + r1_deg * math.cos(bearing - angle_a)
            lon3_2 = lon1 + r1_deg * math.sin(bearing - angle_a)
            
            point1 = (lat3_1, lon3_1)
            point2 = (lat3_2, lon3_2)
            
            # Choose the most likely point
            valid_points = []
            if is_valid_location(point1, buildings):
                valid_points.append(point1)
            if is_valid_location(point2, buildings):
                valid_points.append(point2)
                
            if valid_points:
                return find_best_position(valid_points, buildings[0])
                
        except Exception as e:
            print(f"Error in bilateration calculation: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error in bilateration: {str(e)}")
        return None

def estimate_location_by_trilateration(buildings, distances):
    """Estimate user location using trilateration with improved accuracy"""
    try:
        if len(buildings) != 3 or len(distances) != 3:
            return None

        import numpy as np
        
        # Get coordinates
        points = []
        for building, distance in zip(buildings, distances):
            if building not in BUILDING_COORDINATES:
                return None
            lat, lon = BUILDING_COORDINATES[building]
            points.append((lat, lon, distance))
        
        # Convert to meters for calculation
        earth_radius = 6371000  # Earth's radius in meters
        
        # Convert lat/lon to ECEF coordinates
        ecef_points = []
        for lat, lon, distance in points:
            # Convert to radians
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            
            # Calculate ECEF coordinates
            x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = earth_radius * np.sin(lat_rad)
            
            ecef_points.append((x, y, z, distance))
        
        # Set up matrices for least squares solution
        A = np.zeros((3, 3))
        b = np.zeros(3)
        
        for i in range(3):
            x, y, z, r = ecef_points[i]
            A[i] = [2*(x - ecef_points[2][0]), 
                   2*(y - ecef_points[2][1]), 
                   2*(z - ecef_points[2][2])]
            b[i] = (r**2 - ecef_points[2][3]**2 - 
                   (x**2 + y**2 + z**2) + 
                   (ecef_points[2][0]**2 + 
                    ecef_points[2][1]**2 + 
                    ecef_points[2][2]**2))
        
        try:
            # Solve using least squares
            solution = np.linalg.solve(A, b)
            
            # Convert back to lat/lon
            x, y, z = solution
            lon = np.degrees(np.arctan2(y, x))
            hyp = np.sqrt(x*x + y*y)
            lat = np.degrees(np.arctan2(z, hyp))
            
            if is_valid_location((lat, lon), buildings):
                return (lat, lon)
                
        except np.linalg.LinAlgError:
            return None
            
    except Exception as e:
        print(f"Error in trilateration: {str(e)}")
        return None

def is_valid_location(point, buildings):
    """Check if a location is valid based on building positions"""
    try:
        lat, lon = point
        
        # Check if point is within campus bounds
        min_lat = min(coord[0] for coord in BUILDING_COORDINATES.values()) - 0.001
        max_lat = max(coord[0] for coord in BUILDING_COORDINATES.values()) + 0.001
        min_lon = min(coord[1] for coord in BUILDING_COORDINATES.values()) - 0.001
        max_lon = max(coord[1] for coord in BUILDING_COORDINATES.values()) + 0.001
        
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return False
            
        # Check if point is too close to any building
        for building in BUILDING_COORDINATES:
            b_lat, b_lon = BUILDING_COORDINATES[building]
            distance = calculate_distance_between_points(lat, lon, b_lat, b_lon)
            if distance < 5:  # Minimum 5 meters from any building
                return False
                
        return True
        
    except Exception as e:
        print(f"Error in location validation: {str(e)}")
        return False

def find_best_position(positions, reference_building):
    """Find the most likely position based on campus layout"""
    try:
        if not positions:
            return None
            
        # Calculate scores for each position based on distances to other buildings
        scores = []
        for pos in positions:
            score = 0
            for building, coords in BUILDING_COORDINATES.items():
                if building != reference_building:
                    distance = calculate_distance_between_points(
                        pos[0], pos[1], coords[0], coords[1]
                    )
                    # Prefer positions that are reasonably distant from other buildings
                    if 5 <= distance <= 100:
                        score += 1
            scores.append(score)
            
        # Return the position with the highest score
        if scores:
            return positions[scores.index(max(scores))]
        return positions[0]
        
    except Exception as e:
        print(f"Error in finding best position: {str(e)}")
        return positions[0] if positions else None

def visualize_location_on_map(user_location, building_coordinates, detected_building=None, filename=None, distances=None, building_names=None):
    """Create a map visualization of the user's location and nearby buildings with distances"""
    # Create map centered at user location
    m = folium.Map(location=user_location, zoom_start=18)

    # Add user location marker
    folium.Marker(
        user_location,
        tooltip="Your Location",
        popup="Estimated User Position",
        icon=folium.Icon(color="red", icon="user", prefix="fa")
    ).add_to(m)

    # Add building markers with distances
    for i, (building_name, coords) in enumerate(building_coordinates.items()):
        # Create popup content with building info
        popup_content = f"""
        <b>{building_name}</b><br>
        Coordinates: ({coords[0]:.6f}, {coords[1]:.6f})<br>
        """
        
        # Add distance if available
        if distances and building_names and building_name in building_names:
            idx = building_names.index(building_name)
            if idx < len(distances):
                popup_content += f"Distance: {distances[idx]:.2f} meters"

        # Add building marker
        folium.Marker(
            coords,
            tooltip=building_name,
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color="blue", icon="building", prefix="fa")
        ).add_to(m)

        # Draw line from user to building if it's a detected building
        if building_name in (building_names or []):
            folium.PolyLine(
                locations=[user_location, coords],
                color='green',
                weight=2.5,
                opacity=0.8,
                popup=f"Distance: {distances[building_names.index(building_name)]:.2f} meters"
            ).add_to(m)

    # Add a circle for each distance to show trilateration
    if distances and building_names:
        for i, (building_name, distance) in enumerate(zip(building_names, distances)):
            if building_name in building_coordinates:
                folium.Circle(
                    building_coordinates[building_name],
                    radius=distance,
                    color='green',
                    fill=True,
                    fill_opacity=0.1,
                    popup=f"Distance to {building_name}: {distance:.2f}m"
                ).add_to(m)

    # Add a scale bar
    plugins.Fullscreen().add_to(m)
    folium.plugins.MousePosition().add_to(m)

    # Save map
    if not filename:
        filename = f"map_{uuid.uuid4().hex}.html"
    save_path = os.path.join("static", "maps", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    m.save(save_path)

    return filename

app = Flask(__name__)

@app.route('/process-images', methods=['POST'])
def process_images():
    try:
        model, label_encoder, base_model = load_recognition_model()

        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        images = request.files.getlist('images')
        if len(images) < 1 or len(images) > 3:
            return jsonify({'error': 'Please provide 1 to 3 images'}), 400

        results = []
        building_names = []
        distances = []

        for idx, img_file in enumerate(images):
            # Save image temporarily
            img_path = f'temp_image_{uuid.uuid4().hex}.jpg'
            img_file.save(img_path)

            try:
                # Process image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Get predictions
                features = base_model.predict(img_array)
                prediction = model.predict(features)
                predicted_class = np.argmax(prediction[0])
                building_name = label_encoder.classes_[predicted_class]
                confidence = float(prediction[0][predicted_class])

                # Detect building boundaries and estimate distance
                bbox, pixel_height, img_rgb = detect_building_boundaries(img_path)
                distance = estimate_distance_by_size(building_name, pixel_height) if pixel_height > 0 else None

                result = {
                    'image_index': idx,
                    'building_name': building_name,
                    'confidence': str(confidence),
                    'bounding_box': str(bbox),
                    'pixel_height': str(pixel_height),
                    'distance': str(distance) if distance else 'None'
                }

                building_names.append(building_name)
                distances.append(distance if distance else 0)
                results.append(result)

            finally:
                # Clean up temporary file
                if os.path.exists(img_path):
                    os.remove(img_path)

        # Estimate user location
        user_location = estimate_location(building_names, distances)
        map_filename = None
        map_urls = {
            'local': 'None',
            'accessible': 'None'
        }
        
        if user_location:
            map_filename = visualize_location_on_map(
                user_location, 
                BUILDING_COORDINATES,
                building_names[0] if len(building_names) == 1 else None,
                distances=distances,
                building_names=building_names
            )
            
            # Get the server's IP address
            import socket
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            
            # Create both local and accessible URLs
            map_urls = {
                'local': f"http://127.0.0.1:5000/static/maps/{map_filename}",
                'accessible': f"http://{ip_address}:5000/static/maps/{map_filename}"
            }

        response = {
            'results': results,
            'user_location': str(user_location) if user_location else 'None',
            'map_urls': map_urls,
            'building_coordinates': BUILDING_COORDINATES,
            'distances': [str(d) for d in distances] if distances else [],
            'detected_buildings': building_names
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a route to serve static files
@app.route('/static/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('static/maps', filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(os.path.join('static', 'maps'), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False) 