Okay, I can help you with that. I will use the search tool to get the relevant information from the "Overpass API by Example" page on the OpenStreetMap Wiki and then structure it into a Markdown file format.

```markdown
# Overpass API by Example Summary (for LLM Use)

This document summarizes examples from the OpenStreetMap Wiki page "Overpass API by Example". It aims to provide a structured overview of common Overpass QL query patterns for easy understanding and use by Large Language Models.

**Source:** [https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_API_by_Example](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_API_by_Example)

**Note:** The examples on the wiki page are primarily designed for visualization on [Overpass Turbo](http://overpass-turbo.eu/). This summary focuses on the query structure and purpose.

## Core Concepts

*   **Overpass API:** A read-only API for querying custom selections of OpenStreetMap (OSM) data.
*   **Overpass QL:** The query language used to interact with the Overpass API.
*   **Elements:** OSM data consists of `node` (points), `way` (lines or polygons), and `relation` (groups of elements). `nwr` is a shorthand for node, way, and relation.
*   **Tags:** Key-value pairs describing OSM elements (e.g., `amenity=restaurant`, `highway=bus_stop`).
*   **Statements:** Queries are built from statements that filter data based on location, tags, element types, etc.
*   **Bounding Box (`{{bbox}}`)**: A common way to specify the geographical area for a query, often automatically filled by tools like Overpass Turbo based on the current map view.
*   **Output (`out`)**: Specifies the format and detail level of the results. Common formats include `xml` and `json`. Common detail levels include `out body;` (full data), `out skel;` (ids only), `out geom;` (full geometry), `out center;` (center point for polygons).

## Common Query Examples

*(Note: Query structures are simplified for clarity. Timeout and output format settings like `[out:json][timeout:25];` are often included at the start of real queries.)*

### 1. Finding Specific Features (Tags) in an Area

*   **Purpose:** Find all elements (nodes, ways, relations) with a specific tag within the current map view (bounding box).
*   **Example: Find all restaurants**

    ```ql
    // Find nodes, ways, and relations tagged as restaurants within the bounding box
    nwr["amenity"="restaurant"]({{bbox}});
    // Output the results with geometry
    out geom;
    ```
    *   *(Variation: `out center;` is often used for amenities to get a point marker)*

*   **Example: Find specific named features**

    ```ql
    // Find features named "Somewhere"
    nwr["name"="Somewhere"]({{bbox}});
    out geom;
    ```

### 2. Searching Within a Named Area (Instead of Bounding Box)

*   **Purpose:** Find features within a defined administrative boundary or other named area.
*   **Method:** First, find the area ID, then search within it. `geocodeArea` is a convenient Overpass Turbo extension.
*   **Example: Find pubs in "Leeds"**

    ```ql
    // Define the area to search (Overpass Turbo extension)
    {{geocodeArea:Leeds}}->searchArea;
    // Find pubs (nodes, ways, relations) within that area
    nwr(area.searchArea)["amenity"="pub"];
    // Output results
    out geom;
    ```
    *   *(Note: Without Overpass Turbo, you'd need to find the area's OSM relation ID and use `area(id);`)*

### 3. Combining Multiple Queries (Union)

*   **Purpose:** Find elements matching *any* of several criteria.
*   **Method:** Use a `(` ... `);` block.
*   **Example: Find amenities that are either restaurants OR bars**

    ```ql
    (
      // Find restaurants
      nwr["amenity"="restaurant"]({{bbox}});
      // Find bars
      nwr["amenity"="bar"]({{bbox}});
    );
    // Output the combined results
    out geom;
    ```

### 4. Using Regular Expressions for Tag Values

*   **Purpose:** Find tags where the value matches a pattern (e.g., starts with, contains, case-insensitive).
*   **Method:** Use the tilde `~` operator.
*   **Example: Find amenities starting with "post_" (e.g., post_box, post_office)**

    ```ql
    nwr["amenity"~"^post_"]({{bbox}});
    out geom;
    ```

*   **Example: Find names containing "Street", case-insensitive**

    ```ql
    nwr["name"~"Street",i]({{bbox}}); // 'i' makes it case-insensitive
    out geom;
    ```

### 5. Searching Around a Point (Proximity)

*   **Purpose:** Find features within a certain radius of other features.
*   **Method:** Use the `around` filter.
*   **Example: Find drinking water sources within 100m of picnic sites**

    ```ql
    // Find picnic sites in the bbox and store them
    nwr["tourism"="picnic_site"]({{bbox}}) -> .picnic_sites;
    // Find drinking water nodes within 100m of the found picnic sites
    node(around.picnic_sites:100)["amenity"="drinking_water"];
    // Output the drinking water locations
    out geom;
    ```

### 6. Filtering by Element Type

*   **Purpose:** Search only for specific element types (node, way, or relation).
*   **Example: Find only *ways* tagged as cycleways**

    ```ql
    way["highway"="cycleway"]({{bbox}});
    out geom;
    ```

### 7. Counting Elements

*   **Purpose:** Get a count of features matching criteria instead of the features themselves.
*   **Example: Count addr:housenumber elements in an area (CSV output)**

    ```ql
    [out:csv(::count,::count(nodes),::count(ways),::count(relations))];
    // Define the area
    {{geocodeArea:YourCityName}}->.a;
    // Find elements with housenumbers in the area
    (
      node["addr:housenumber"](area.a);
      way["addr:housenumber"](area.a);
      relation["addr:housenumber"](area.a);
    );
    // Output the counts
    out count;
    ```

### 8. Working with Relations (Recursion)

*   **Purpose:** Find elements that are members of a relation, or relations that contain certain elements.
*   **Method:** Use recursion operators like `>`, `<`, `>>`, `<<`.
*   **Example: Get all nodes belonging to ways tagged as footpaths**

    ```ql
    way["highway"="footpath"]({{bbox}});
    // Recurse down to get the nodes that make up these ways
    >;
    out skel; // Often use 'skel' (skeleton) for just IDs and geometry
    ```

## Further Learning

*   **Overpass Turbo:** [http://overpass-turbo.eu/](http://overpass-turbo.eu/) (Interactive query builder and map)
*   **Overpass API Language Guide:** [https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide](https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide)
*   **Overpass QL Reference:** [https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL)

This summary provides a foundation based on the "Overpass API by Example" page. The Overpass language offers many more complex features for advanced data extraction.
```