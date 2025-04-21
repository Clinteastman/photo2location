import { useState, ChangeEvent, FormEvent, useEffect } from 'react';
import './App.css';
// Import Leaflet CSS
import 'leaflet/dist/leaflet.css';

// Import components from react-leaflet
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap
} from 'react-leaflet';
import L, { LatLngExpression } from 'leaflet';

// --- Fix for default Leaflet marker icon --- 
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});
// --- End of fix ---

// Define the structure of the expected result from the backend
interface GeolocationResult {
  lat: number;
  lon: number;
  tags: Record<string, string>;
  score?: number;
}

// Component to automatically adjust map view when results change
interface MapUpdaterProps {
  results: GeolocationResult[];
}

function MapUpdater({ results }: MapUpdaterProps) {
  const map = useMap();
  useEffect(() => {
    if (results.length > 0) {
      const bounds = L.latLngBounds(results.map(r => [r.lat, r.lon]));
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [50, 50] });
      }
    } else {
      map.setView([51.505, -0.09], 2);
    }
  }, [results, map]);

  return null;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<GeolocationResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [vlmDescription, setVlmDescription] = useState<string | null>(null);
  
  // New state variables for Auto Run Query feature
  const [autoRunQuery, setAutoRunQuery] = useState<boolean>(true);
  const [customQuery, setCustomQuery] = useState<string>('');
  const [showQueryView, setShowQueryView] = useState<boolean>(false);

  // Handler for auto run toggle
  const handleAutoRunToggle = (e: ChangeEvent<HTMLInputElement>) => {
    setAutoRunQuery(e.target.checked);
  };

  // Handler for custom query change
  const handleQueryChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setCustomQuery(e.target.value);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setResults([]);
      setVlmDescription(null);
      setCustomQuery('');
      setShowQueryView(false);

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
    }
  };

  // Handler for manual query execution
  const executeQuery = async () => {
    if (!customQuery) {
      setError('Please provide a valid Overpass query.');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Execute the custom query
      const response = await fetch('http://localhost:8000/api/execute_query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: customQuery })
      });

      if (!response.ok) {
        let errorMsg = `Error: ${response.status} ${response.statusText}`;
        try {
          const errorData = await response.json();
          errorMsg = errorData.detail || errorMsg;
        } catch { /* Ignore if response not JSON */ }
        throw new Error(errorMsg);
      }

      const data: GeolocationResult[] = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Query execution failed:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setError('Please select an image file first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults([]);
    setVlmDescription(null);
    setCustomQuery('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Add auto_run parameter to request
    const url = autoRunQuery 
      ? 'http://localhost:8000/api/geolocate' 
      : 'http://localhost:8000/api/analyze_only';

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = `Error: ${response.status} ${response.statusText}`;
        try {
          const errorData = await response.json();
          errorMsg = errorData.detail || errorMsg;
        } catch { /* Ignore if response not JSON */ }
        throw new Error(errorMsg);
      }

      if (autoRunQuery) {
        // Direct result mode - just get and display locations
        const data: GeolocationResult[] = await response.json();
        setResults(data);
      } else {
        // Analysis only mode - get the generated query
        const data = await response.json();
        setCustomQuery(data.query);
        setVlmDescription(data.description || null);
        setShowQueryView(true);
      }

    } catch (err) {
      console.error('Geolocation request failed:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <h1>Photo2Location</h1>
      <p>Upload an image to find its potential location.</p>

      <form onSubmit={handleSubmit}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <button type="submit" disabled={!selectedFile || isLoading}>
            {isLoading ? 'Analyzing...' : 'Geolocate Image'}
          </button>
          
          {/* Auto Run Query toggle */}
          <label style={{ display: 'flex', alignItems: 'center', marginLeft: '20px' }}>
            <input 
              type="checkbox" 
              checked={autoRunQuery} 
              onChange={handleAutoRunToggle} 
              style={{ marginRight: '5px' }} 
            />
            Auto Run Query
          </label>
        </div>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {showQueryView && !autoRunQuery && (
        <div style={{ marginTop: '20px' }}>
          <h3>Generated Overpass Query:</h3>
          <div style={{ display: 'flex', gap: '10px', flexDirection: 'column' }}>
            <textarea 
              value={customQuery} 
              onChange={handleQueryChange} 
              style={{ 
                width: '100%', 
                height: '200px', 
                fontFamily: 'monospace',
                padding: '10px',
                border: '1px solid #ccc'
              }}
            />
            <div>
              <button 
                onClick={executeQuery} 
                disabled={isLoading}
                style={{ padding: '8px 16px' }}
              >
                {isLoading ? 'Running...' : 'Run Query'}
              </button>
              <span style={{ fontSize: '0.9em', marginLeft: '10px' }}>
                (Modify the query if needed before running)
              </span>
            </div>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', marginTop: '20px', gap: '20px' }}>
        {/* Image Preview Area */}
        <div style={{ flex: 1 }}>
          {previewUrl && (
            <div>
              <h2>Preview:</h2>
              <img src={previewUrl} alt="Selected preview" style={{ maxWidth: '100%', maxHeight: '400px', border: '1px solid #ccc' }} />
            </div>
          )}
          {vlmDescription && (
            <div style={{ marginTop: '10px' }}>
              <h3>AI Description:</h3>
              <p><i>{vlmDescription}</i></p>
            </div>
          )}
        </div>

        {/* Map and Results Area */}
        <div style={{ flex: 2 }}>
          <h2>Potential Locations Map</h2>
          {isLoading && <p>Loading results...</p>}

          <MapContainer center={[51.505, -0.09]} zoom={2} style={{ height: '500px', width: '100%' }}>
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <MapUpdater results={results} />
            {results.map((result, index) => {
              const position: LatLngExpression = [result.lat, result.lon];
              const googleStreetViewUrl = `https://www.google.com/maps?q&layer=c&cbll=${result.lat},${result.lon}`;
              const osmUrl = `https://www.openstreetmap.org/?mlat=${result.lat}&mlon=${result.lon}#map=17/${result.lat}/${result.lon}`;
              const mapillaryUrl = `https://www.mapillary.com/app/?lat=${result.lat}&lng=${result.lon}&z=17`;

              return (
                <Marker key={index} position={position}>
                  <Popup>
                    <b>Candidate {index + 1}</b> (Score: {result.score?.toFixed(2) ?? 'N/A'})<br />
                    Lat: {result.lat.toFixed(5)}, Lon: {result.lon.toFixed(5)}<br />
                    <b>Tags:</b><br />
                    <pre style={{ maxHeight: '100px', overflowY: 'auto', background: '#f0f0f0', padding: '5px' }}>
                      {JSON.stringify(result.tags, null, 2)}
                    </pre>
                    <b>Verify:</b><br />
                    <a href={googleStreetViewUrl} target="_blank" rel="noopener noreferrer">Google Street View</a><br />
                    <a href={osmUrl} target="_blank" rel="noopener noreferrer">OpenStreetMap</a><br />
                    <a href={mapillaryUrl} target="_blank" rel="noopener noreferrer">Mapillary</a>
                  </Popup>
                </Marker>
              );
            })}
          </MapContainer>
        </div>
      </div>
    </>
  );
}

export default App;
