from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
import csv
from io import StringIO

# Add the statisticaldrafting module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'statisticaldrafting'))

import statisticaldrafting as sd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def parse_collection_string(collection_str):
    """
    Parse a comma-separated string of double-quoted card names.
    
    Examples:
    - '"Lightning Bolt","Llanowar Elves"' -> ['Lightning Bolt', 'Llanowar Elves']
    - '"Serra\'s Angel","Birds of Paradise"' -> ["Serra's Angel", "Birds of Paradise"]
    - '"Card, with comma","Another card"' -> ["Card, with comma", "Another card"]
    """
    if not collection_str or collection_str.strip() == '':
        return []
    
    try:
        # Use CSV parser to handle quoted strings with commas properly
        reader = csv.reader(StringIO(collection_str))
        parsed = next(reader)
        return [card.strip() for card in parsed if card.strip()]
    except Exception as e:
        # Fallback: try simple comma split if CSV parsing fails
        try:
            # Remove quotes and split by comma
            cleaned = collection_str.replace('"', '').replace("'", "'")
            return [card.strip() for card in cleaned.split(',') if card.strip()]
        except Exception:
            raise ValueError(f"Failed to parse collection string: {collection_str}. Use format: '\"Card Name\",\"Another Card\"'")

@app.route('/pick-order', methods=['GET'])
def get_pick_order():
    """
    Get pick order for a given collection.
    
    Query parameters:
    - set: MTG set code (optional, defaults to "DFT")
    - draft_mode: Draft mode (optional, defaults to "Premier")
    - collection: Comma-separated, double-quoted list of card names (optional, defaults to empty for P1P1)
    
    Examples:
    - /pick-order (Pack 1 Pick 1)
    - /pick-order?collection="Lightning Bolt","Llanowar Elves"
    - /pick-order?set=BLB&collection="Courageous Goblin","Serra's Angel"
    """
    try:
        # Get parameters from query string
        set_code = request.args.get('set', 'DFT')
        draft_mode = request.args.get('draft_mode', 'Premier')
        collection_str = request.args.get('collection', '')
        
        # Parse collection string
        collection = parse_collection_string(collection_str)
        
        # Create draft model
        dm = sd.DraftModel(set=set_code, draft_mode=draft_mode)
        
        # Get pick order
        pick_order = dm.get_pick_order(collection)
        
        # Convert to JSON-friendly format
        pick_order_json = pick_order.to_dict('records')
        
        return jsonify({
            "set": set_code,
            "draft_mode": draft_mode,
            "collection": collection,
            "collection_count": len(collection),
            "pick_order": pick_order_json
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found. Use GET /pick-order"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting MTG Statistical Drafting API...")
    print("Endpoint: GET /pick-order")
    print("  Parameters:")
    print("    - set: MTG set code (optional, defaults to 'DFT')")
    print("    - draft_mode: Draft mode (optional, defaults to 'Premier')")
    print("    - collection: Comma-separated, double-quoted card names (optional)")
    print()
    print("Example usage:")
    print("  http://localhost:5000/pick-order")
    print("  http://localhost:5000/pick-order?set=FDN&draft_mode=Premier")
    print('  http://localhost:5000/pick-order?collection="Lightning Bolt","Llanowar Elves"')
    print('  http://localhost:5000/pick-order?set=FDN&collection="Serra\'s Angel","Birds of Paradise"')
    print()
    
    # Check if models directory exists
    if not os.path.exists('data/models'):
        print("WARNING: data/models directory not found!")
        print("Make sure you have trained models in the data/models directory.")
    else:
        model_files = [f for f in os.listdir('data/models') if f.endswith('.pt')]
        print(f"Found {len(model_files)} model files in data/models/")
    
    print("Starting server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 