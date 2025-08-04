import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def plotGenes(geneDataList, timePoints, plotTitle = None, plotLegend = None, 
showPoints = False, plotColors = None, lineOpacity = 1, axis = None,
xAxisLabel = "Time (hours)", yAxisLabel = "Gene expression", plotGrid = True):
  """
  Plots the temporal trajectories of a given set of genes, smoothed using cubic
  interpolating splines.
  
  geneDataList: Length-N list of N genes' time series data (each represented as 
  a list), each having n measurements over time.
  
  timePoints: Length-n array listing the hours corresponding to each time point.
  
  plotTitle: Optional plot title.
  
  plotLegend: Optional length-N list of names to display as labels for each gene.
  
  showPoints: True or False indicating whether observed data points should be 
  plotted atop their corresponding spline interpolants.
  
  plotColors: Optional length-N list of colors to use for each line (gene).
    
  lineOpacity: Numeric opacity from 0 to 1 of plotted lines. Default = 1 (opaque).
  
  axis: Optional matplotlib axes to draw the plot on. If None, a new figure and 
  axes will be created.
  
  xAxisLabel, yAxisLabel: Optional labels for the x- and y-axes. 
  
  plotGrid: True or False indicating whether to include grid lines
  """
  
  createdNewAxis = False
  
  if axis is None:
    fig, axis = plt.subplots()
    createdNewAxis = True
  else:
    fig = axis.figure
    
  # Set grid lines if desired
  if plotGrid:
    axis.grid(alpha = 0.5)
  
  for index, data in enumerate(geneDataList):
    # Create the spline interpolant
    spline = CubicSpline(timePoints, data, bc_type = "natural")
    xPoints = np.linspace(np.min(timePoints), np.max(timePoints), 500)
    yPoints = spline(xPoints)
    
    # Choose line color, depending on whether or not it is provided
    lineColor = plotColors[index] if plotColors else None

    # Draw the interpolant
    plotLabel = plotLegend[index] if plotLegend else None
    line, = axis.plot(xPoints, yPoints, label = plotLabel, color = lineColor, alpha = lineOpacity)
    
    # Add points at timePoints if desired
    if showPoints:
      axis.scatter(timePoints, data, color = line.get_color(), s = 15, alpha = lineOpacity)
  
  # Set plot title if provided
  if plotTitle:
    axis.set_title(plotTitle)

  # Set legend if labels are provided
  if plotLegend:
    axis.legend()
    
  # Set x and y axis labels
  axis.set_xlabel(xAxisLabel)
  axis.set_ylabel(yAxisLabel)
  
  # Adjust spacing of plot elements
  fig.tight_layout()
  
  # Display plot if new axes were created, or return the plot object if not
  # (i.e. if the current plot is going to be a subplot of a larger plot)
  if createdNewAxis:
    plt.show()
  else:
    return fig
