# generate_pdf_report
An example script demonstrating how to automatically generate a formatted .pdf report. 
In this example I am using the generic output of a binary classification model and conducting a few simple calculations to assess the model performance. Refer to the generate_report function for an example on to generate a .pdf report. The example dataset contains 100,000 datapoints, but any similar dataset can be used.

The pandas, numpy, matplotlib, and reportlab libraries are required for the code to be operational, and the most recent version of pandas is required (0.25.x) since I used some newly released features to analyze the model output data. Once these libraries are installed and up to date simply running the file will generate a text summary of my analysis, a .jpeg of the ROC curve, and a .pdf report with a complete summary of the analysis.

To test this project, simply download and run the .py file after the appropriate libraries have been installed.
