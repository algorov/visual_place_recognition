import folium

def plot_results(coords):
    m = folium.Map(location=coords[0], zoom_start=16)
    for lat, lon in coords:
        folium.Marker(location=[lat, lon]).add_to(m)
    m.save("map.html")