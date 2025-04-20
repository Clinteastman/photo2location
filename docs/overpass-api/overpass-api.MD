# OpenStreetMap (OSM) Overpass API Documentation Summary

## 1. Introduction: What is the Overpass API?

*   The Overpass API is a **read-only** Application Programming Interface (API) designed for **querying** and retrieving custom selections of OpenStreetMap (OSM) map data. [1]
*   It acts like a database accessible over the web: you send a query, and it returns the OSM data (nodes, ways, relations) that matches your criteria. [1, 8]
*   It is optimized for **data consumers** needing specific subsets of OSM data, selected by criteria like location, object type, tag properties, proximity, or combinations thereof. [1, 4]
*   Unlike the main OSM editing API, the Overpass API is not intended for map editing. [1, 10]
*   It keeps a copy of the main OSM database, updated frequently (often with minutely diffs), allowing for relatively up-to-date queries. [8, 10]

## 2. Key Features & Capabilities

*   **Filtering:** Select data based on tags (key-value pairs), location (bounding boxes, polygons, around a point), element type (nodes, ways, relations), proximity, and more complex combinations. [1, 4, 5]
*   **Data Selection:** Retrieve specific OSM elements (nodes, ways, relations) matching the query. [1]
*   **Flexibility:** Supports complex queries to find specific features, check data consistency, or gather data for specialized maps. [4, 3]
*   **Read-Only Access:** Designed specifically for querying, not modifying, OSM data. [1]

## 3. Query Languages

The Overpass API supports two main query languages:

1.  **Overpass QL (Query Language):**
    *   The recommended, more modern, and concise language. [5, 9]
    *   Features a C-style syntax, procedural and imperative. [9, 18]
    *   Statements end with a semicolon (`;`). [5, 9, 18]
    *   Core statements include `node`, `way`, `relation` (or `rel`), `nwr` (for node, way, and relation), recursion operators (`<`, `>`), and output statements (`out`). [5, 9]
2.  **Overpass XML:**
    *   The original query language, using XML syntax. [5, 9]
    *   More verbose than Overpass QL.
    *   Queries can often be converted between QL and XML. [5]

## 4. Accessing the API

*   **Endpoints:** Queries are sent via HTTP GET or POST requests to an API endpoint URL. [15]
    *   **Main Public Instance:** `https://overpass-api.de/api/interpreter` [1]
    *   **Other Public Instances:** Several alternatives exist (e.g., `maps.mail.ru/osm/tools/overpass/api/interpreter`, `overpass.osm.ch/api/interpreter`, `overpass.private.coffee/api/interpreter`, `overpass.openstreetmap.ru/api/interpreter`). [1]
    *   **Recommended Endpoint Path:** Use `/api/interpreter` for Overpass QL/XML queries. [7, 26] Avoid the older `/api/xapi` path. [26]
*   **Sending Queries:**
    *   **GET Requests:** Shorter queries can be appended to the endpoint URL, typically after `?data=`. The query string might need URL encoding. [15, 14, 21]
        ```
        https://overpass-api.de/api/interpreter?data=[out:json];node(around:100,51.5,-0.1)[amenity=cafe];out;
        ```
    *   **POST Requests:** Preferred for longer or more complex queries. The query is sent in the request body. [15]
    *   **Command Line:** Tools like `curl` or `wget` can be used to send queries. [14, 15]
        ```bash
        # Using GET
        wget -O cafes.json "https://overpass-api.de/api/interpreter?data=[out:json];node(around:100,51.5,-0.1)[amenity=cafe];out;"

        # Using POST (query in file query.ql)
        echo "[out:json];node(around:100,51.5,-0.1)[amenity=cafe];out;" > query.ql
        wget -O cafes.json --post-file=query.ql "https://overpass-api.de/api/interpreter"
        ```

## 5. Output Formats

*   The API can return data in various formats, specified using `[out:...]` at the beginning of the query or within the `out` statement. [5]
*   Common formats include:
    *   **JSON:** `[out:json];` (Often preferred for web applications)
    *   **XML:** `[out:xml];` (Default if not specified)
    *   **CSV:** `[out:csv(...)];` (Requires specifying columns)
    *   **Custom formats:** Possible via configuration.

## 6. Overpass Turbo: The Interactive Frontend

*   **Overpass Turbo (`overpass-turbo.eu`)** is a highly recommended web-based tool for working with the Overpass API. [1, 2, 3]
*   **Features:**
    *   Write and execute Overpass QL/XML queries. [2, 11]
    *   Visualize results interactively on a map. [2, 3, 11]
    *   Includes a **Wizard** to help beginners construct queries from simpler search terms. [2, 3, 25]
    *   Export query results (e.g., GeoJSON, GPX, KML). [1, 2, 20]
    *   Export the query itself (e.g., as a URL for the API). [26]
    *   Excellent for testing, debugging, and developing queries. [3, 8, 13]

## 7. Usage Policies and Limits

*   Public Overpass API instances are **shared resources**. Users should avoid excessive load. [4, 16]
*   **Rate Limiting:** Servers enforce limits to ensure fair usage. [16]
*   **Resource Limits:** Queries have default limits on execution time (e.g., 180 seconds) and memory usage (e.g., 512 MiB). These can often be increased using query settings like `[timeout:...]` and `[maxsize:...]`, but server maximums still apply. [16]
*   **Fair Use Guidelines (e.g., for `overpass-api.de`):** Aim for less than 10,000 queries per day and download less than 1 GB per day as a rough guideline. [1]
*   **Heavy Usage:** For very demanding applications, consider setting up your own Overpass instance or using a commercial provider (like Geofabrik). [4, 17]
*   **Identify Your Application:** When using the API programmatically, provide a descriptive `User-Agent` HTTP header identifying your application. [10]

## 8. Basic OSM Data Model Concepts

Understanding the basic OSM data elements helps in writing queries: [21, 18]

*   **Nodes:** Represent points in space (latitude, longitude). Can represent point features (like benches, trees) or be part of ways. [21]
*   **Ways:** Ordered lists of nodes. Represent linear features (like roads, rivers) or areas (like buildings, parks - if the first and last nodes are the same). [21]
*   **Relations:** Group other elements (nodes, ways, or other relations) together to represent complex features (like bus routes, administrative boundaries, multipolygons). [21]
*   **Tags:** Key-value pairs attached to nodes, ways, or relations that describe the feature (e.g., `amenity=restaurant`, `highway=residential`, `name=Main Street`). [21, 18]

## 9. Client Libraries

Libraries are available for various programming languages to simplify interaction with the Overpass API, handling HTTP requests, and parsing responses:

*   **Python:** `overpass` library [6, 21], direct `requests` usage.
*   **JavaScript/Node.js:** Libraries like `query-overpass`, `overpass-frontend`, direct `fetch`/`axios` usage. [1]
*   **Others:** Check language-specific package repositories. [1, 24]

## 10. Where to Learn More

*   **Overpass API Wiki Main Page:** [https://wiki.openstreetmap.org/wiki/Overpass_API](https://wiki.openstreetmap.org/wiki/Overpass_API) [1]
*   **Overpass QL Language Reference:** [https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) [9, 5, 12]
*   **Overpass API Language Guide (Examples & Concepts):** [https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide](https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide) [5]
*   **Overpass API by Example:** [https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_API_by_Example](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_API_by_Example) [27]
*   **Overpass Turbo Wiki Page:** [https://wiki.openstreetmap.org/wiki/Overpass_turbo](https://wiki.openstreetmap.org/wiki/Overpass_turbo) [3]
*   **Overpass API User's Manual:** [https://dev.overpass-api.de/overpass-doc/en/](https://dev.overpass-api.de/overpass-doc/en/) [12, 22]
*   **Overpass Turbo Instance:** [https://overpass-turbo.eu/](https://overpass-turbo.eu/) [2, 11]