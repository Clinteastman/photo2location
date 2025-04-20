# Plan of Action: Image Geolocation Web App via AI Analysis and Overpass API

## 1. Project Goal

To develop a web-based application where a user uploads an image, an AI analyzes the image to identify geographically relevant features (landmarks, building types, text, environment), generates queries for the OpenStreetMap Overpass API based on this analysis, and presents potential geographic locations on a map, allowing the user to manually verify results using integrated links to street-level imagery services.

## 2. Core Concept

Image Input -> AI Analysis (Feature & Type Identification) -> Overpass Query Generation -> Overpass API Execution -> Location Candidate Ranking -> Web UI Display (Map + Verification Links)

## 3. Detailed Workflow

1.  **Image Input (Frontend):**
    *   User accesses the web application.
    *   User uploads an image file via a file selector or drag-and-drop interface.
    *   Frontend displays an image preview and initiates the backend process.

2.  **AI Image Analysis (Backend):**
    *   Backend receives the image file.
    *   Backend sends the image to a Vision-Language Model (VLM) API (e.g., GPT-4V, Gemini Pro Vision).
    *   **Prompt Engineering:** The prompt directs the VLM to:
        *   Identify prominent landmarks, buildings, natural features.
        *   Suggest the *type* of each feature (e.g., "church", "bridge", "tower", "office building", "park").
        *   Map these types to potential OpenStreetMap (OSM) tags (e.g., `amenity=place_of_worship`, `man_made=bridge`).
        *   Extract any visible text (OCR) from signs, banners, etc.
        *   Describe the general environment (urban, rural, coastal, mountainous).
        *   Suggest potential region/country hints if possible.
    *   **Output:** The VLM returns a structured format (preferably JSON) containing the description, identified features with type suggestions and potential OSM tags, extracted text, and environment/region hints.

3.  **Overpass Query Generation (Backend):**
    *   Backend parses the structured JSON output from the VLM.
    *   **Strategy:** Translate the identified features, types, potential tags, and their implied spatial relationships into Overpass QL (Query Language).
        *   Prioritize specific names extracted via OCR (e.g., `way["name"="Rue de la Paix"]`).
        *   Search for combinations of identified feature types/tags in proximity (e.g., `node[amenity=place_of_worship](around:500, way[leisure=park]);`).
        *   Use region hints to add `area[...]` filters or `bbox` constraints to limit the search scope.
        *   Consider using another LLM call specifically tasked with generating Overpass QL from the VLM's JSON output, or use a rule-based system mapping feature types/tags to query patterns.
    *   **Output:** One or more Overpass QL query strings.

4.  **Overpass API Execution (Backend):**
    *   Backend sends the generated Overpass QL query(s) to a public or private Overpass API endpoint via HTTP request.
    *   Handle potential timeouts, errors, rate limits, and large responses.

5.  **Results Processing & Ranking (Backend):**
    *   Backend receives the Overpass API response (typically OSM data in JSON format).
    *   Parse the OSM data (nodes, ways, relations) representing potential location candidates.
    *   **Ranking Logic:** Score candidates based on:
        *   Number of distinct VLM-identified features matched in the OSM data.
        *   Proximity of matched features in the candidate location.
        *   Match quality (e.g., exact name match > tag match).
        *   Consistency with negative constraints (e.g., if VLM said "no water visible").
    *   Filter out irrelevant or low-scoring results.

6.  **Presentation & Manual Verification (Frontend):**
    *   Backend sends the ranked list of top N candidate locations (coordinates, score, supporting OSM info) back to the frontend.
    *   Frontend displays the candidates as markers on an interactive map (e.g., using Leaflet).
    *   Frontend also displays the original AI description for context.
    *   **Crucial Step:** For each candidate marker, provide direct links that open in a new tab:
        *   **Google Street View:** Link to the specific lat/lon (`https://www.google.com/maps?q&layer=c&cbll=[LAT],[LON]...`).
        *   **Mapillary:** Link to view imagery at the coordinates, if available.
        *   **(Optional) Google Maps 3D/Satellite View.**
        *   **OpenStreetMap.org:** Link to the location on OSM.
    *   The user clicks these links to visually compare the street-level view with their original image to confirm or reject the AI's suggestions.

## 4. Key Components & Technologies

*   **Frontend:**
    *   Framework: React, Vue, Svelte, Angular (or plain HTML/CSS/JS for simpler MVP)
    *   Mapping Library: Leaflet (recommended), Mapbox GL JS, OpenLayers
    *   HTTP Client: `fetch` API, axios
*   **Backend:**
    *   Language/Framework: Python (Flask/FastAPI/Django), Node.js (Express)
    *   HTTP Client: `requests` (Python), `axios`/`node-fetch` (Node.js)
    *   Libraries: `overpass` (Python, optional), JSON handling.
*   **External Services:**
    *   Vision AI: OpenAI API (GPT-4V), Google Cloud Vision / Gemini API, Anthropic Claude API.
    *   Overpass API: Public instance (e.g., `overpass-api.de`) or self-hosted instance.
    *   Mapping Services (for verification links): Google Maps, Mapillary.

## 5. Major Challenges & Considerations

*   **Ambiguity:** Many locations look similar; VLM descriptions might be generic.
*   **VLM Accuracy:** Model might miss details, hallucinate, or fail to recognize specific landmarks/types. Output quality is critical.
*   **Description-to-Query Gap:** Translating fuzzy natural language/JSON descriptions into precise, effective Overpass QL is complex.
*   **OSM Data:** Location might exist but not be mapped, or mapped inaccurately/outdatedly in OpenStreetMap.
*   **Overpass Query Performance:** Broad or complex queries can be slow or time out. Need for efficient query design and potential area constraints.
*   **Scalability & Cost:** AI API calls cost money. High traffic may require backend scaling and potentially a dedicated Overpass instance.
*   **Verification Necessity:** The system provides *candidates*; final confirmation relies on user's manual verification via linked services (Street View, etc.). Manage expectations.
*   **Error Handling:** Implement robust handling for API errors, invalid inputs, no results found.

## 6. User Interface (UI) / User Experience (UX) Considerations

*   Simple and intuitive image upload process.
*   Clear visual feedback during processing (loading indicators).
*   Interactive map display that is easy to navigate.
*   Clear presentation of candidate locations and their scores/details.
*   Easily accessible and clearly labeled verification links (Street View, Mapillary).
*   Display the AI's textual description alongside the map for reference.

## 7. Next Steps (Initial Phase - MVP)

1.  Select specific technologies (Frontend framework, Backend framework, VLM provider).
2.  Set up basic project structure (Frontend app, Backend API).
3.  Implement image upload functionality (Frontend -> Backend).
4.  Integrate with the chosen VLM API to get image analysis (Backend).
5.  Develop initial logic for generating a simple Overpass query based on VLM output (e.g., using the most confident landmark or name).
6.  Integrate with the Overpass API to execute the query (Backend).
7.  Implement basic result display on a Leaflet map (Frontend).
8.  Add static verification links (Google Street View) for each result marker (Frontend).
9.  Test with clear, well-known landmarks first. Iterate and refine query generation and ranking logic.