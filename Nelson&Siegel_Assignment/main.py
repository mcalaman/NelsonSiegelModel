# imports 
import os
import sys
import pandas as pd


# import classes
subfolder_path = os.path.join(os.getcwd(), "helpers")
sys.path.append(subfolder_path)
import NelsonSiegelScript

# examples of different objects created 
my_investment_universe = NelsonSiegelScript.InvestmentUniverse()
my_benchmark = NelsonSiegelScript.Benchmark()
my_alpha_index = NelsonSiegelScript.AlphaStrategy()
my_portfolio = NelsonSiegelScript.Portfolio()

# Examples of objects created

# Output n 1
# my_investment_universe.get_charts_amount_outstanding()

# Output n 2 (see Output n. 5 for more details)
# my_investment_universe.opportunity_set

# Output n 3
# my_benchmark.index_cumulative_returns

# Output n 4
# my_alpha_index.model_goodness_fit

# Output n 5
# my_portfolio.performance_output
