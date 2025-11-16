from flask import Blueprint, request, jsonify

dss_bp = Blueprint('dss_bp', __name__)

@dss_bp.route('/recommend', methods=['POST'])
def recommend_schemes():
    """
    Decision Support System (DSS) endpoint.
    Receives land cover statistics and recommends government schemes.
    """
    try:
        data = request.json
        recommendations = []
        
        # Get the stats calculated by the frontend
        agri_percent = data.get('agriculture_percent', 0)
        water_percent = data.get('water_percent', 0)
        forest_percent = data.get('forest_percent', 0)

        # --- This is your DSS Logic ---
        # Rule for Jal Jeevan Mission
        if water_percent < 5:
            recommendations.append({
                "scheme": "Jal Jeevan Mission",
                "reason": f"Water bodies cover only {water_percent:.1f}% of the area. Recommending water conservation and supply scheme."
            })
            
        # Rule for PM-KISAN
        if agri_percent > 10:
            recommendations.append({
                "scheme": "PM-KISAN",
                "reason": f"Significant agricultural land ({agri_percent:.1f}%) detected. Recommending farmer income support."
            })
        
        # Rule for National Afforestation Programme
        if forest_percent < 15 and agri_percent > 5:
             recommendations.append({
                "scheme": "National Afforestation Programme",
                "reason": f"Forest cover is low ({forest_percent:.1f}%). Recommending agroforestry and plantation on farm bunds."
            })

        if len(recommendations) == 0:
             recommendations.append({
                "scheme": "No specific schemes triggered",
                "reason": "The area meets baseline criteria."
            })
            
        return jsonify(recommendations)

    except Exception as e:
        print(f"Error in DSS: {e}")
        return jsonify({"error": str(e)}), 400