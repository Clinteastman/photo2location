from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv
import logging
import asyncio
import json # Added import
import base64 # Added import
from io import BytesIO # Added import
import re # Added import
import pycountry # Added import for ISO code lookup

# Added imports for Google Gemini
import google.generativeai as genai
from PIL import Image

from pydantic import BaseModel, ValidationError
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS Configuration ---
# Allow requests from your frontend development server
# TODO: Restrict origins in production
origins = [
    "http://localhost:5173", # Default Vite dev server port
    "http://127.0.0.1:5173",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Pydantic Models (Data Validation) ---
class VLMFeature(BaseModel):
    description: str
    type: Optional[str] = None
    potential_osm_tags: Optional[List[str]] = None

class VLMAnalysis(BaseModel):
    general_description: str
    features: List[VLMFeature]
    extracted_text: Optional[str] = None
    environment_hint: Optional[str] = None
    region_hint: Optional[str] = None

class OverpassResult(BaseModel):
    lat: float
    lon: float # Changed from dict to float
    tags: dict # Added missing tags field
    score: Optional[float] = None # Placeholder for ranking score

# --- Placeholder Functions (To be implemented) ---

async def analyze_image_with_vlm(image_bytes: bytes) -> VLMAnalysis:
    """
    Sends image bytes to the Google Gemini API for analysis.
    """
    logger.info("Analyzing image with Google Gemini...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        raise HTTPException(status_code=500, detail="VLM API key not configured.")

    try:
        genai.configure(api_key=api_key)

        # Load image from bytes
        img = Image.open(BytesIO(image_bytes))

        # Select the model
        # Use gemini-1.5-flash for potentially faster/cheaper results, or gemini-pro-vision
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Define the prompt, asking for JSON output matching VLMAnalysis structure
        prompt = f"""
Analyze the provided image and describe its geographically relevant features.
Identify prominent landmarks, buildings, natural features, and any visible text (OCR).
Suggest the type for each feature (e.g., 'church', 'bridge', 'park').
Map these types to potential OpenStreetMap (OSM) tags (e.g., `amenity=place_of_worship`, `man_made=bridge`).
Describe the general environment (e.g., urban, rural, coastal).
Suggest a potential region or country if possible.

Return the analysis STRICTLY in the following JSON format, matching the provided schema. Do not include any text outside the JSON block:

```json
{{
  "general_description": "<Overall description of the scene>",
  "features": [
    {{
      "description": "<Description of feature 1>",
      "type": "<Suggested type, e.g., building, river, sign>",
      "potential_osm_tags": ["<tag1=value1>", "<tag2=value2>"]
    }},
    {{
      "description": "<Description of feature 2>",
      "type": "<Suggested type>",
      "potential_osm_tags": ["<tag3=value3>"]
    }}
    // ... more features
  ],
  "extracted_text": "<Any text extracted via OCR>",
  "environment_hint": "<e.g., urban, rural, mountainous, coastal>",
  "region_hint": "<Suggested country or region>"
}}
```
"""

        logger.info("Sending request to Gemini API...")
        # Send image and prompt to the model
        response = await model.generate_content_async([prompt, img])

        logger.info("Received response from Gemini API.")
        # Extract the JSON part of the response
        try:
            # Gemini might wrap the JSON in ```json ... ``` or just return it.
            json_text = response.text
            if json_text.strip().startswith("```json"):
                json_text = json_text.strip()[7:-3].strip()
            elif json_text.strip().startswith("```"):
                 json_text = json_text.strip()[3:-3].strip()

            logger.debug(f"Raw JSON text from Gemini: {json_text}")
            analysis_data = json.loads(json_text)

            # Validate the received data against the Pydantic model
            vlm_analysis = VLMAnalysis(**analysis_data)
            logger.info("Successfully parsed and validated Gemini response.")
            return vlm_analysis

        except (json.JSONDecodeError, ValidationError, AttributeError, IndexError) as e:
            logger.error(f"Failed to parse or validate Gemini response: {e}")
            logger.error(f"Raw Gemini response text: {response.text}")
            raise HTTPException(status_code=500, detail=f"Failed to process VLM response: {e}")
        except Exception as e:
             logger.error(f"Unexpected error processing Gemini response: {e}")
             logger.error(f"Raw Gemini response text: {response.text}")
             raise HTTPException(status_code=500, detail=f"Unexpected error processing VLM response: {e}")


    except Exception as e:
        logger.error(f"Error calling Google Gemini API: {e}", exc_info=True)
        # Check for specific Gemini API errors if the library provides them
        raise HTTPException(status_code=502, detail=f"VLM API request failed: {e}")

def generate_overpass_query(analysis: VLMAnalysis) -> str:
    """
    Generates an Overpass QL query string based on VLM analysis.
    Combines searches for individual names and relevant feature tags.
    Adds region hint if available using ISO 3166-1 country codes.
    """
    logger.info("Generating Overpass query...")
    
    # Start query with JSON output format and reasonable timeout
    query_parts = ["[out:json][timeout:60];"]
    clauses = []
    area_filter = "" # Initialize area filter
    
    # --- Strategy 0: Use Region Hint for Area Filter ---
    logger.debug(f"Checking region hint: '{analysis.region_hint}' (Type: {type(analysis.region_hint)})")
    if analysis.region_hint:
        # Strip whitespace and clean up the hint
        hint_clean = analysis.region_hint.strip()
        hint_lower = hint_clean.lower()
        
        # Try to find the country ISO code using pycountry or direct mapping
        try:
            # First try exact name match
            country = None
            try:
                country = pycountry.countries.get(name=hint_clean)
            except (AttributeError, LookupError):
                pass
            
            # Try with lowercase
            if not country:
                try:
                    country = pycountry.countries.get(name=hint_lower)
                except (AttributeError, LookupError):
                    pass
            
            # Try lookup by common words
            if not country:
                # Dictionary of keywords to ISO codes
                country_keywords = {
                    'uk': 'GB', 
                    'united kingdom': 'GB',
                    'britain': 'GB',
                    'england': 'GB',
                    'usa': 'US',
                    'united states': 'US',
                    'america': 'US',
                    # Add more common aliases as needed
                }
                
                for keyword, iso in country_keywords.items():
                    if keyword in hint_lower:
                        try:
                            country = pycountry.countries.get(alpha_2=iso)
                            break
                        except (AttributeError, LookupError):
                            continue
            
            # If we found a country, use its alpha-2 code for the area filter
            if country and hasattr(country, 'alpha_2'):
                country_code = country.alpha_2
                country_name = country.name
                logger.info(f"Found ISO code '{country_code}' for region hint '{hint_clean}'")
                
                # Add the area search and set the filter
                query_parts.insert(1, f'area["ISO3166-1"="{country_code}"]->.searchArea;')
                area_filter = "(area.searchArea)"
                logger.info(f"Applying area filter for: {country_name}")
            else:
                logger.warning(f"Could not map region hint '{hint_clean}' to a known ISO country code. Query will be global.")
        except Exception as e:
            # In case of any errors in country detection, log and continue without area filter
            logger.error(f"Error looking up country code for '{hint_clean}': {e}")
            logger.warning("Could not determine country code from region hint. Query will be global.")
    else:
        logger.warning("No region hint provided. Query will be global.")

    # --- Strategy 1: Add Clauses for Individual Extracted Names ---
    potential_names = []
    if analysis.extracted_text:
        # Expanded list of generic terms to ignore
        generic_terms = {
            "give way", "stop", "exit", "entrance", "warning", "caution", 
            "open", "closed", "push", "pull", "emergency", "information"
        }
        
        # Split by comma and filter out generic terms
        potential_names = [
            name.strip() for name in analysis.extracted_text.split(',')
            if name.strip() and name.strip().lower() not in generic_terms and len(name.strip()) > 2
        ]

    if potential_names:
        logger.info(f"Adding query clauses for potential names: {potential_names}")
        for name in potential_names:
            # Use the shortened 'nwr' syntax instead of separate node/way/relation queries
            # Use case-insensitive regex match with proper escaping
            escaped_name = re.escape(name)
            clauses.append(f'nwr["name"~"{escaped_name}",i]{area_filter};')
    
    # --- Strategy 2: Add Clauses for Relevant Feature Tags ---
    # Define feature types we want to query for (expanded based on OSM tag patterns)
    relevant_feature_types = {
        'church', 'building', 'shops', 'historic', 'landmark', 
        'restaurant', 'hotel', 'tourism', 'amenity'
    }
    added_tags = set()  # Track added tags to avoid duplicates

    if analysis.features:
        logger.info("Adding query clauses for relevant feature tags.")
        for feature in analysis.features:
            # Check if feature type is relevant and has tags
            if feature.type and feature.potential_osm_tags:
                # Check if the feature type is one we're interested in
                feature_type_relevant = feature.type.lower() in relevant_feature_types
                
                # Process tags if the feature type is relevant or if we don't have many clauses yet
                if feature_type_relevant or len(added_tags) < 5:
                    for tag in feature.potential_osm_tags:
                        # Basic tag validation
                        if '=' in tag:
                            # Normalize tag for duplicate checking
                            normalized_tag = tag.lower().strip()
                            if normalized_tag not in added_tags:
                                key, value = tag.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                if key and value:
                                    # Use nwr shorthand for cleaner queries
                                    clauses.append(f'nwr["{key}"="{value}"]{area_filter};')
                                    added_tags.add(normalized_tag)
                                    logger.debug(f"Added tag query: {key}={value}")

    # --- Combine clauses and finalize query ---
    if not clauses:
        logger.warning("Could not generate any query clauses from VLM analysis.")
        return ""  # Indicate no query could be generated
    
    # Use a union block to combine all clauses (match any)
    query_parts.append("(")
    
    # Limit to a reasonable max number of clauses to avoid overloading the API
    max_clauses = min(len(clauses), 10)
    query_parts.extend([f"  {clauses[i]}" for i in range(max_clauses)])
    
    query_parts.append(");")
    
    # Add output format - center points work well for our use case
    query_parts.append("out center;")
    
    # Join everything into a final query string
    query = "\n".join(query_parts)
    
    logger.debug(f"Generated Overpass Query:\n{query}")
    return query

def execute_overpass_query(query: str) -> List[OverpassResult]:
    """
    Executes the given Overpass QL query against the configured API endpoint.
    Parses the results into a list of OverpassResult objects.
    """
    if not query:
        logger.warning("execute_overpass_query called with an empty query string. Returning empty list.")
        return []

    overpass_endpoint = os.getenv("OVERPASS_API_ENDPOINT", "https://overpass-api.de/api/interpreter")
    logger.info(f"Executing Overpass query against {overpass_endpoint}...\nQuery:\n{query}") # Log the query being sent

    response = None # Initialize response to None
    try:
        response = requests.post(
            overpass_endpoint,
            data=query,
            headers={'Content-Type': 'application/sparql-query'}, # Technically Overpass QL, but this often works
            timeout=300 # Set a request timeout (seconds)
        )

        # Check for HTTP errors
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        logger.info(f"Overpass query executed successfully (Status: {response.status_code}). Parsing results...")
        data = response.json()

        results: List[OverpassResult] = []
        if "elements" in data:
            for element in data["elements"]:
                lat, lon = None, None
                tags = element.get("tags", {})

                if element["type"] == "node":
                    lat = element.get("lat")
                    lon = element.get("lon")
                elif element["type"] in ["way", "relation"]:
                    # If 'out center;' was used, coordinates are in 'center'
                    center = element.get("center")
                    if center:
                        lat = center.get("lat")
                        lon = center.get("lon")
                    # Fallback: sometimes nodes within ways/relations might be returned,
                    # but 'out center;' is the primary way to get a single point.
                    # We could potentially try to find a representative node if center is missing,
                    # but for now, we rely on 'out center;'.

                if lat is not None and lon is not None:
                    try:
                        result = OverpassResult(lat=lat, lon=lon, tags=tags)
                        results.append(result)
                    except ValidationError as ve:
                        logger.warning(f"Skipping element due to validation error: {ve}. Element data: {element}")
                else:
                     logger.debug(f"Skipping element without valid coordinates: Type={element.get('type')}, ID={element.get('id')}")

        logger.info(f"Parsed {len(results)} results from Overpass response.")
        return results

    except requests.exceptions.Timeout:
        logger.error(f"Overpass API request timed out after 300 seconds.")
        raise HTTPException(status_code=504, detail="Overpass API request timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error executing Overpass query: {e}")
        # Try to get more specific error from response if available
        status_code = 502 # Bad Gateway default
        detail = f"Overpass API request failed: {e}"
        # Use e.response which is guaranteed to exist for HTTPError subclasses
        # For other RequestException, response might be None
        current_response = e.response if hasattr(e, 'response') else response
        if current_response is not None:
            status_code = current_response.status_code
            try:
                # Overpass might return error details in HTML or plain text
                error_detail = current_response.text[:500] # Limit error message size
                detail = f"Overpass API error (Status {status_code}): {error_detail}"
            except Exception:
                pass # Keep the original detail
        raise HTTPException(status_code=status_code, detail=detail)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from Overpass API: {e}")
        # Safely access response text if response exists
        raw_text = response.text[:500] if response else "[No response object]"
        logger.error(f"Raw response text: {raw_text}...") # Log beginning of response
        raise HTTPException(status_code=502, detail="Failed to decode Overpass API response.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Overpass query execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during Overpass query: {e}")

def rank_results(results: List[OverpassResult], analysis: VLMAnalysis) -> List[OverpassResult]:
    """
    Ranks Overpass results based on matching VLM analysis features and text.
    Assigns a score to each result and sorts the list descending by score.
    """
    logger.info(f"Ranking {len(results)} results based on VLM analysis...")

    if not results:
        return []

    # --- Scoring Weights (can be tuned) ---
    TEXT_MATCH_SCORE = 5.0       # Score for matching extracted text (e.g., in name tag)
    TAG_MATCH_SCORE = 1.0        # Score for each matching OSM tag from VLM features
    # Add more sophisticated scoring later (e.g., proximity, environment match)

    # --- Prepare VLM data for easier lookup ---
    vlm_text = analysis.extracted_text.strip().lower() if analysis.extracted_text else None
    vlm_tags = set()
    for feature in analysis.features:
        if feature.potential_osm_tags:
            for tag in feature.potential_osm_tags:
                if '=' in tag:
                    vlm_tags.add(tag.lower()) # Store tags as lowercase key=value strings

    # --- Calculate score for each result ---
    for result in results:
        score = 0.0
        result_tags_lower = {k.lower(): v.lower() for k, v in result.tags.items()}

        # 1. Check for Text Match (primarily in 'name' tag)
        if vlm_text:
            # Check name tag specifically
            if result_tags_lower.get('name') == vlm_text:
                score += TEXT_MATCH_SCORE
                logger.debug(f"Result {result.lat},{result.lon}: Text match on name tag ('{vlm_text}') +{TEXT_MATCH_SCORE}")
            else:
                # Check if text appears in any other tag value (lower score?)
                # For now, keep it simple and focus on name
                pass

        # 2. Check for Tag Matches
        matched_vlm_tags = set()
        for res_key, res_value in result_tags_lower.items():
            res_tag_str = f"{res_key}={res_value}"
            if res_tag_str in vlm_tags:
                # Avoid double-counting if VLM provided the same tag multiple times
                if res_tag_str not in matched_vlm_tags:
                    score += TAG_MATCH_SCORE
                    logger.debug(f"Result {result.lat},{result.lon}: Tag match ('{res_tag_str}') +{TAG_MATCH_SCORE}")
                    matched_vlm_tags.add(res_tag_str)

        result.score = score
        logger.debug(f"Result {result.lat},{result.lon}: Final score = {score}")

    # --- Sort results by score (descending) ---
    sorted_results = sorted(results, key=lambda r: r.score or 0, reverse=True)

    logger.info(f"Ranking complete. Top result score: {sorted_results[0].score if sorted_results else 'N/A'}")
    return sorted_results
    # --- End of function ---

# --- API Endpoint ---

@app.post("/api/geolocate", response_model=List[OverpassResult])
async def geolocate_image(file: UploadFile = File(...)):
    """
    Main endpoint to receive an image, analyze it, query Overpass, and return ranked results.
    """
    logger.info(f"Received image: {file.filename}, content type: {file.content_type}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")

        # 1. AI Image Analysis
        vlm_analysis = await analyze_image_with_vlm(image_bytes)
        logger.info(f"VLM Analysis Result: {vlm_analysis}")

        # 2. Overpass Query Generation
        overpass_query = generate_overpass_query(vlm_analysis)
        logger.info(f"Generated Overpass Query: {overpass_query}")

        # 3. Overpass API Execution
        overpass_results = execute_overpass_query(overpass_query)
        logger.info(f"Overpass Results (Raw): {overpass_results}")

        # 4. Results Processing & Ranking
        ranked_results = rank_results(overpass_results, vlm_analysis)
        logger.info(f"Ranked Results: {ranked_results}")

        return ranked_results

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"An error occurred during geolocation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        await file.close()

# After the existing geolocate_image endpoint
@app.post("/api/analyze_only")
async def analyze_image_only(file: UploadFile = File(...)):
    """
    Endpoint that only analyzes the image and returns the generated query without executing it.
    Used when auto_run is disabled in the frontend.
    """
    logger.info(f"Received image for analysis only: {file.filename}, content type: {file.content_type}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")

        # 1. AI Image Analysis
        vlm_analysis = await analyze_image_with_vlm(image_bytes)
        logger.info(f"VLM Analysis Result: {vlm_analysis}")

        # 2. Overpass Query Generation
        overpass_query = generate_overpass_query(vlm_analysis)
        logger.info(f"Generated Overpass Query: {overpass_query}")

        # Return the query and VLM description, but don't execute it
        return {
            "query": overpass_query,
            "description": vlm_analysis.general_description,
            "extracted_text": vlm_analysis.extracted_text,
            "region_hint": vlm_analysis.region_hint
        }

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        await file.close()

# Add execute_query endpoint
@app.post("/api/execute_query", response_model=List[OverpassResult])
async def execute_custom_query(query_request: dict):
    """
    Endpoint that executes a custom Overpass query provided by the user.
    Used when the user modifies the generated query in the frontend.
    """
    if not query_request.get("query"):
        raise HTTPException(status_code=400, detail="Missing query parameter")
    
    query = query_request.get("query")
    logger.info(f"Executing custom query: {query}")

    try:
        # Execute the provided query
        overpass_results = execute_overpass_query(query)
        logger.info(f"Custom query returned {len(overpass_results)} results")
        
        # We don't have VLM analysis for ranking, so return as-is
        # Could potentially store the last VLM analysis in a cache/session for ranking
        return overpass_results

    except Exception as e:
        logger.error(f"Error executing custom query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing query: {e}")

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "Photo2Location Backend is running!"}

# --- How to Run (Instructions for development) ---
# 1. Make sure you are in the 'backend' directory.
# 2. Activate the virtual environment: source venv/bin/activate
# 3. Run the server: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#    - `--reload`: Automatically restarts the server when code changes.
#    - `--host 0.0.0.0`: Makes the server accessible on your network.
#    - `--port 8000`: Specifies the port to run on.

# --- Next Steps from project-outline.MD (MVP) ---
# [ ] 3. Implement image upload functionality (Frontend -> Backend) - Basic API endpoint created.
# [X] 4. Integrate with the chosen VLM API - Placeholder added. Needs actual implementation.
# [X] 5. Develop initial logic for generating Overpass query - Placeholder added. Needs actual implementation.
# [X] 6. Integrate with the Overpass API - Placeholder added. Needs actual implementation.
# [ ] 7. Implement basic result display on Leaflet map (Frontend).
# [ ] 8. Add static verification links (Frontend).
# [ ] 9. Test and Iterate.

# --- TODO ---
# - Implement actual VLM API call in analyze_image_with_vlm.
# - Implement robust Overpass query generation in generate_overpass_query.
# - Implement actual Overpass API call and parsing in execute_overpass_query.
# - Implement ranking logic in rank_results.
# - Add proper error handling for API calls.
# - Secure API keys using environment variables (.env).
# - Configure CORS more restrictively for production.
