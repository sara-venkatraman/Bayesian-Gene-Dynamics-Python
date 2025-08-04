import pandas as pd
import numpy as np
import itertools
from scipy import integrate

from scipy.interpolate import CubicSpline
import time
from joblib import Parallel, delayed


def LLR2(geneDataList, timePoints, bayes, priorMatrix = None, writeToCSV = True):
  """
  geneDataList: Length-N list of N genes' time series data (each represented as 
  a list), each having n measurements over time.
  
  timePoints: Length-n array listing the times at which measurements were taken.
  
  bayes: True or False, indicating whether or not the Bayesian method should be 
  used to calculate the LLR2 values.
  
  priorMatrix: (Bayesian method only) N-by-N dataframe giving a prior adjacency 
  matrix, whose (i,j) entry is 1 if the two corresponding genes are likely to be 
  associated, 0 if unlikely, and missing if unknown.
  
  writeToCSV: True or False indicating whether the LLR2 matrix should be written
  to a CSV file in the current working directory.
  """
  
  # Store number of genes
  nGenes = len(geneDataList)
  
  # --- Integration ---
  
  # Define upper and lower bounds of integrals that will be computed
  lowerBounds = timePoints[:-1]
  upperBounds = timePoints[1:]
  
  # Calculate the integral lists for all genes
  integrals = [integrateGeneData(g, timePoints, lowerBounds, upperBounds) for g in geneDataList]
  
  # --- Compute R^2 value from all regressions ---
  
  # Get the indices of the gene pairs that will be used in each regression
  genePairIndices = list(itertools.combinations(range(len(geneDataList)), 2))
  
  # Compute LLR2 between all gene pairs and print timing
  startTime = time.time()
  if bayes:
    LLR2 = Parallel(n_jobs = -1)(delayed(computeLLR2Bayes)(geneDataList, priorMatrix, timePoints, integrals, i, j) for i, j in genePairIndices)
  else:
    LLR2 = Parallel(n_jobs = -1)(delayed(computeLLR2OLS)(geneDataList, timePoints, integrals, i, j) for i, j in genePairIndices)
  endTime = time.time()
  elapsedTime = endTime - startTime
  print(f"Runtime: {elapsedTime:.4f} seconds")
  
  # --- Convert list of R^2 values to a symmetric similarity matrix ---

  # Create empty matrix and fill upper triangle
  R2Matrix = np.zeros((nGenes, nGenes))
  for (i, j), value in zip(genePairIndices, LLR2):
    R2Matrix[i, j] = value
  
  # Repeat values in lower triangle for symmetry
  R2Matrix = R2Matrix + R2Matrix.T
  
  # Fill in diagonal
  np.fill_diagonal(R2Matrix, 1)
  
  # Save LLR2 matrix to CSV, if desired
  if writeToCSV:
    filename = "BayesLLR2.csv" if bayes else "OLSLLR2.csv" 
    np.savetxt(filename, R2Matrix, delimiter = ",")


# === Helper functions ===

def integrator(f, lower, upper):
  """
  Computes the definite integral of a function f from lower to upper.
  """
  return integrate.quad(f, lower, upper)[0]


def integrateGeneData(geneDataValues, timePoints, lowerBounds, upperBounds):
  """
  Defines a cubic spline interpolant from a gene's expression values and 
  integrates it between all corresponding pairs of bounds.
  """
  # Create the spline interpolant
  spline = CubicSpline(timePoints, geneDataValues, bc_type = "natural")
  
  # Integrate spline between all pairs of bounds
  integrals = [integrator(spline, lower, upper) for lower, upper in zip(lowerBounds, upperBounds)]
  
  # Return list of integrals with 0 prepended
  return [0] + integrals


def getDesignMatrix(geneDataList, timePoints, integrals, indexA, indexB):
  """
  Construct the design matrix for a single regression.
  """
  x = pd.DataFrame({"mB": geneDataList[indexB], "int_mB": integrals[indexB], "int_mA": integrals[indexA], "time": timePoints, "intercept": 1})
  return x


def computeLLR2Bayes(geneDataList, priorMatrix, timePoints, integrals, indexA, indexB):
  """
  Computes the Bayesian LLR2 value for the two genes at indices indexA and
  indexB within geneDataList.
  """
  # Define X and Y for regression of gene A on gene B
  x1 = getDesignMatrix(geneDataList, timePoints, integrals, indexA, indexB)
  y1 = geneDataList[indexA]
  
  # Define dimensions
  n, p = x1.shape
  
  # Define X and Y for regression of gene B on gene A
  x2 = pd.DataFrame({"mB": y1, "int_mB": x1["int_mA"], "int_mA": x1["int_mB"], "time": timePoints, "intercept": 1})
  y2 = x1["mB"]

  # Set prior mean of regression coefficients
  prior = priorMatrix.iloc[indexA, indexB]
  if pd.isnull(prior):
    priorMean = [0] * 5
  else:
    priorMean = [int(prior > 0)] * 2 + [0] * 3
    
  # Compute main LLR2 value, first direction
  LScoefs = np.linalg.lstsq(x1, y1, rcond = None)[0]
  LSfit = np.array(x1 @ LScoefs)
  if pd.isnull(prior) or prior > 0:
    sigmaSq = np.linalg.norm(y1 - LSfit) ** 2 / (n - p)
    g = (np.linalg.norm(LSfit - (x1 @ priorMean)) ** 2 - p * sigmaSq) / (p * sigmaSq) 
  else:
    g = 1
    
  posteriorMean = (1 / (1 + g)) * np.array(priorMean) + (g / (1 + g)) * np.array(LScoefs)
  posteriorFit = x1 @ posteriorMean
  LLR2_dir1 = np.var(posteriorFit, ddof = 1) / (np.var(posteriorFit, ddof = 1) + np.var(y1 - posteriorFit, ddof = 1))

  # Compute main LLR2 value, second direction
  LScoefs = np.linalg.lstsq(x2, y2, rcond = None)[0] # LLR2model_dir2.coef_
  LSfit = np.array(x2 @ LScoefs) # LLR2model_dir2.predict(x2)
  if pd.isnull(prior) or prior > 0:
    sigmaSq = np.linalg.norm(y2 - LSfit) ** 2 / (n - p)
    g = (np.linalg.norm(LSfit - (x2 @ priorMean)) ** 2 - p * sigmaSq) / (p * sigmaSq) 
  else:
    g = 1
    
  posteriorMean = (1/(1 + g)) * np.array(priorMean) + (g/(1 + g)) * np.array(LScoefs)
  posteriorFit = x2 @ posteriorMean
  LLR2_dir2 = np.var(posteriorFit, ddof = 1) / (np.var(posteriorFit, ddof = 1) + np.var(y2 - posteriorFit, ddof = 1))
  
  # Return the larger of the two LLR2 values
  return max(LLR2_dir1, LLR2_dir2)


def computeLLR2OLS(geneDataList, timePoints, integrals, indexA, indexB):
  """
  Computes the non-Bayesian (least squares) LLR2 value for the two genes at 
  indices indexA and indexB within geneDataList.
  """
  # Define X and Y for regression of gene A on gene B
  x1 = getDesignMatrix(geneDataList, timePoints, integrals, indexA, indexB)
  y1 = geneDataList[indexA]

  # Define X and Y for regression of gene B on gene A
  x2 = pd.DataFrame({"mB": y1, "int_mB": x1["int_mA"], "int_mA": x1["int_mB"], "time": timePoints, "intercept": 1})
  y2 = x1["mB"]

  # Compute main LLR2 value, first direction
  LScoefs = np.linalg.lstsq(x1, y1, rcond = None)[0]
  LSfit = np.array(x1 @ LScoefs)
  MSS = np.sum(LSfit ** 2)
  RSS = np.sum((y1 - LSfit) ** 2)
  LLR2_dir1 = MSS / (MSS + RSS)

  # Compute main LLR2 value, second direction
  LScoefs = np.linalg.lstsq(x2, y2, rcond = None)[0]
  LSfit = np.array(x2 @ LScoefs)
  MSS = np.sum(LSfit ** 2)
  RSS = np.sum((y2 - LSfit) ** 2)
  LLR2_dir2 = MSS / (MSS + RSS)

  # Return the larger of the two LLR2 values
  return max(LLR2_dir1, LLR2_dir2)
  
