DEPI Graduation Project
Natural gas pipelines Buildings Violation Auto Detection

Introduction:
According to Egyptian law No.4 for year 1988 which regulates the Natural gas pipelines and petroleum pipelines, it’s forbidden to perform any building activities near pipelines within 6m boundary in desert and agriculture areas. And within 2m in cities and urban areas.
And with the rapid advance in the special imaging which provides high accuracy imaging using satellites for all geographic locations.
It’s possible to develop a model that can detect these violations automatically and generate an automated alarm to alert the responsible people to take the required action to stop this violation.
Project Design:
•	Expected inputs: 
o	Natural gas pipeline coordinates in the form of CSV list.
o	Satellite imaging using Sentinel-2 Satellite Images.
o	Type of area (rural or urban)
o	Accepted date range for satellite images history.
•	Expected outputs: 
o	List of detected violations with coordinates and the minimum distance from pipeline and date of image shows the violation as the date of construction.
o	Graphical presentation showing the pipeline and detected violations.

Project phases:
1-	Preparing the Pipelines training data.
2-	Preparing the satellite images training data
3-	Preparing the model for violations detection.
4-	Preparing the GUI.
5-	Testing the model.
6-	Comparing the output results with the real world locations. 

