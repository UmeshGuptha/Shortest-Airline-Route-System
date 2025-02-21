from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#Goroka,Sochi international
airport=pd.read_csv("Full_Merge_of_All_Unique Airports.csv")
airline=pd.read_csv("Updated_Airline_Routes.csv")

airline.pop("Airline ID")
airline=airline.drop_duplicates(ignore_index=True)
app = Flask(__name__)

import heapq
def create_graph():
    global airline
    g={}
    for i in range(0,airline["Departure"].count()):
        if airline._get_value(i,"Departure") not in g:
            g[airline._get_value(i,"Departure")]=[(airline._get_value(i,"Destination"),airline._get_value(i,"distance"))]
        else:
            g[airline._get_value(i,"Departure")].append((airline._get_value(i,"Destination"),airline._get_value(i,"distance")))
    return g
def uniform_cost_search(graph, start, goal):
    visited = set()
    frontier = []

    # Starting node
    heapq.heappush(frontier, (0, start, []))

    while frontier:
        (cost, current, path) = heapq.heappop(frontier)

        if current not in visited:
            visited.add(current)
            path = path + [current]

            if current == goal:
                return path, cost
            if current in graph.keys():
                for neighbor, edge_cost in graph[current]:
                    heapq.heappush(frontier, (cost + edge_cost, neighbor, path))

    return None, None
def world_map(path,source_airport,destination_airport,cost):
    fig = plt.figure(figsize=(15, 10))
    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')

    
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    
    def get_coordinates(location):
        lon = airport[airport['ID'] == location]['Longitude'].values[0]
        lat = airport[airport['ID'] == location]['Latitude'].values[0]
        return lat, lon
    x2=0
    y2=0
    for i in range(len(path) - 1):
        lat1, lon1 = get_coordinates(path[i])
        lat2, lon2 = get_coordinates(path[i + 1])
        x1, y1 = m(lon1, lat1)
        x2, y2 = m(lon2, lat2)
        
        m.drawgreatcircle(lon1,lat1,lon2,lat2,color='green',linewidth=1)
        m.plot(x1, y1, 'bo', markersize=3, label=path[i])
    m.plot(x2, y2, 'bo', markersize=3, label=path[len(path) - 1])

    
    x1,y1=get_coordinates(path[0])
    x1,y1=m(y1,x1)
    plt.text(x1, y1, path[0], fontsize=8, ha='right',color='red')

    
    plt.text(x2, y2, path[-1], fontsize=8, ha='left',color='red')

    plt.title(f'{source_airport.title()} to {destination_airport.title()},Total Distance:{round(cost,2)}KM')
    plt.legend()
    plt.show()

def check_for_path(source_airport, destination_airport):
    if "airport" not in source_airport.lower():
        source_airport+=" airport"
    if "airport" not in destination_airport.lower():
        destination_airport+=" airport"
    src_id=airport[airport["Label"].str.lower()==source_airport.lower()]["ID"]
    des_id=airport[airport["Label"].str.lower()==destination_airport.lower()]["ID"]
    if not src_id.empty and not des_id.empty:
        print(src_id,des_id)
        g=create_graph()
        path,cost = uniform_cost_search(g, src_id.values[0],des_id.values[0])
        if path is None:
            output_data="No path found from {} to {}".format(source_airport,destination_airport)
        else:
            output_data=f"The shortest Route from {source_airport.title()} to {destination_airport.title()} is: {path},Distance:{round(cost,2)}KM"
            world_map(path,source_airport,destination_airport,cost) 
    elif src_id.empty:
        output_data="Airport with {} is not present".format(source_airport.title())
    else:
        output_data="Airport with {} is not present".format(destination_airport.title())

    return output_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_form', methods=['POST'])
def process_form():
    try:
        source_airport = (request.form['source_airport'])
        destination_airport = (request.form['destination_airport'])
        # Add your processing logic here
        output_data = check_for_path(source_airport, destination_airport)
        if "not present" in output_data:
            raise Exception
        # Render the same template with the form and processed data
        return render_template('index.html', source_airport=source_airport, destination_airport=destination_airport, output_data=output_data)
    except Exception:
        error_message = "Please enter valid airports."
        return render_template('index.html', output_data=output_data,error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)


