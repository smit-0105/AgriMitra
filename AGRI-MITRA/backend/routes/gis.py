from flask import Blueprint, request, jsonify
import numpy as np

gis_bp = Blueprint('gis_bp', __name__)

@gis_bp.route('/assets/<claim_id>', methods=['GET'])
def get_gis_assets(claim_id):
    """
    Serves the asset data (elevation and land cover) for the 3D map.
    Currently uses MOCK data.
    In a real app, you would load a GeoTIFF or database here based on claim_id.
    """
    print(f"Loading GIS assets for claim: {claim_id}")
    
    try:
        # --- MOCK DATA GENERATION ---
        width = 200
        height = 200
        
        # Generate complex elevation data with NumPy
        x = np.linspace(-5, 5, width)
        y = np.linspace(-5, 5, height)
        xx, yy = np.meshgrid(x, y)
        
        elevation = (np.sin(xx**2 + yy**2) * 10 +  # Main valley
                     np.sin(xx * 5) * np.cos(yy * 5) * 5 + # Ridges
                     np.random.rand(width, height) * 2) # Noise
        
        elevation_flat = elevation.flatten().tolist()
        
        # Generate mock land cover data (Classes 0-7)
        landcover = np.zeros((height, width), dtype=np.uint8)
        
        landcover[elevation > 10] = 2  # Dense Forest (Class 2)
        landcover[elevation < -5] = 1  # Water (Class 1)
        landcover[np.logical_and(elevation > 0, elevation < 5)] = 3 # Agriculture (Class 3)
        landcover[np.logical_and(elevation > 5, elevation < 10)] = 6 # Sparse Veg (Class 6)
        landcover[np.logical_and(elevation > -5, elevation < 0)] = 5 # Bare Soil (Class 5)
        
        # Add a "village" (Built-up - Class 4)
        village_center = (height // 2, width // 2)
        landcover[village_center[0]-5:village_center[0]+5, 
                  village_center[1]-5:village_center[1]+5] = 4
        
        landcover_flat = landcover.flatten().tolist()
        
        # --- END MOCK DATA ---

        return jsonify({
            "claim_id": claim_id,
            "width": width,
            "height": height,
            "elevation": elevation_flat, # List of elevation values
            "landcover": landcover_flat  # List of land cover class values (0-7)
        })

    except Exception as e:
        print(f"Error generating GIS assets: {e}")
        return jsonify({"error": str(e)}), 500