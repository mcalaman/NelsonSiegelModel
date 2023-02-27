import pandas as pd
import datetime 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import os


class InvestmentUniverse:
    """
    This class is used to load the data and perform the data cleaning process
    Call the method get_charts_amount_outstanding() to see the histogram and box-plot of the variable amount outstanding
    """
    def __init__(self):
        
        # Loading the data
        input_file_path = os.path.join(os.getcwd(), "Data", 'Data_extract.csv')
        self.data = pd.read_csv(input_file_path)
        
        # Formatting dates and yields
        self.list_asof_dates = list(self.data.ASOF_DATE.unique())
        self.list_asof_dates.sort()
        self.formatting_raw_data()
        
        # Estimating missing MTD Returns
        self.new_issuances = self.identify_new_issuances()
        self.estimate_mtd_return()
        
        # Treatment of Bonds that matured during the month
        self.matured_bonds = self.identify_matured_bonds()
        self.clean_data_matured_bonds()
        
        # Applying the Appropriate filters to define the opportunity set
        self.opportunity_set = self.get_opportunity_set()
        self.opportunity_set_stats = self.get_opportunity_set_stats()
        
        # Charts
        self.histogram_amt_out = None
        self.box_plot_amt_out = None
        
        # Print Output
        self.print_output1()
        
    def formatting_raw_data(self):
        """
        Method for basic formatting
        """
        # Formatting dates
        self.data[['ASOF_DATE','ISSUEDATE','MATURITYDATE']] = self.data[['ASOF_DATE','ISSUEDATE','MATURITYDATE']].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
        # Formatting yields
        self.data[['ACCRUEDINTEREST','COUPON','YIELDTOWORST', 'TOTALRETURNMTD']] /= 100
        # Creating a key identifier for each pair ISIN as ASOF_DATE
        self.data["temp_identifier"] = (self.data.ASOF_DATE.dt.date).astype("str") + "_" + self.data.ISIN
        
    def identify_new_issuances(self):
        """
        Method used to identify bonds issued within the same month as ASOF_DATE
        """
        return self.data.loc[(self.data.TOTALRETURNMTD == 0) &
                       (self.data.ASOF_DATE.dt.month == self.data.ISSUEDATE.dt.month),["ISIN","ASOF_DATE"]].set_index("ISIN").to_dict()["ASOF_DATE"]
    
    def identify_matured_bonds(self):
        """
        Method used to identify bonds matured within the same month as ASOF_DATE
        """
        return self.data.loc[(min(self.list_asof_dates) < self.data.MATURITYDATE) &
                       (self.data.MATURITYDATE <= max(self.list_asof_dates)) & 
                       (self.data.MODDURTOWORST == 0), ["ISIN","ASOF_DATE"]].set_index("ISIN").to_dict()["ASOF_DATE"]
    
    def estimate_mtd_return(self):
        """
        This method is used to estimate the yield of newly issued bonds
        """
        # iterative process for each new issuance 
        for ISIN_, date_ in self.new_issuances.items(): 
            # selecting the relevant date
            df_2 = self.data.loc[(self.data.ASOF_DATE == date_)].copy()
            df_2.reset_index(drop = True, inplace = True)
            # duration of the bond for which we want to estimate returns  
            target_mod_dtw = float(df_2.loc[(df_2.ISIN ==ISIN_),"MODDURTOWORST"])
            # fraction of the month for which the bond has been live
            fraction_month_bond_live = float((df_2.loc[(df_2.ISIN ==ISIN_), "ASOF_DATE"].dt.day - df_2.loc[(df_2.ISIN ==ISIN_), "ISSUEDATE"].dt.day)/df_2.loc[(df_2.ISIN ==ISIN_), "ASOF_DATE"].dt.day)
            # bond with the closest duration
            index_similar_bond = abs(df_2.loc[(df_2.ISIN !=ISIN_) & (df_2.TOTALRETURNMTD != 0), ["MODDURTOWORST"]] - target_mod_dtw).idxmin()[0]
            # return (in bps) of the bond with the closest duration
            mdt_ret_in_bps = float(df_2.loc[(df_2.index == index_similar_bond), "TOTALRETURNMTD"] / df_2.loc[(df_2.index == index_similar_bond), "MODDURTOWORST"])
            # modified duration bond with missing data
            mod_dtw_bond = float(df_2.loc[(df_2.ISIN == ISIN_), "MODDURTOWORST"])
            # computing the mtd_return for the bond with missing data
            self.data.loc[((self.data.ISIN == ISIN_) & (self.data.ASOF_DATE == date_)), "TOTALRETURNMTD"] = mod_dtw_bond * mdt_ret_in_bps * fraction_month_bond_live
    
    def clean_data_matured_bonds(self):
        """
        Method used to fill missing data for matured bonds
        """
        # Using the latest available data to fill gaps
        for ISIN_, date_ in self.matured_bonds.items():      
            self.data.loc[((self.data.ISIN == ISIN_) & (self.data.ASOF_DATE == date_)), "COUPON"] = (self.data.loc[((self.data.ISIN == ISIN_) & (self.data.ASOF_DATE != date_)), "COUPON"].unique()[0])
            self.data.loc[((self.data.ISIN == ISIN_) & (self.data.ASOF_DATE == date_)), "COUPONFREQ"] = (self.data.loc[((self.data.ISIN == ISIN_) & (self.data.ASOF_DATE != date_)), "COUPONFREQ"].unique()[0])
            self.data.loc[((self.data.ISIN == ISIN_) & (self.data.ASOF_DATE == date_)), "COUPONTYPE"] = (self.data.loc[((self.data.ISIN == ISIN_) & (self.data.ASOF_DATE != date_)), "COUPONTYPE"].unique()[0])
            self.data.loc[((self.data.ISIN == ISIN_) & (self.data.ASOF_DATE == date_)), "ACCRUEDINTEREST"] = 0
    
    def get_opportunity_set(self):
        """
        Method used to apply the relevant filters and derive the opportunity set
        """
        # Time remaining until maturity: MATURITYDATE - ASOF_DATE, must be between 1-20 years
        # Time since Issuance: ASOF_DATE – ISSUEDATE, must be no greater than 10 years
        return self.data.loc[((self.data.MATURITYDATE - (self.data.ASOF_DATE + pd.offsets.DateOffset(years = 20))).dt.days < 0) &
                      ((self.data.MATURITYDATE - (self.data.ASOF_DATE + pd.offsets.DateOffset(years = 1))).dt.days > 0) &
                      ((self.data.ASOF_DATE - (self.data.ISSUEDATE + pd.offsets.DateOffset(years = 10))).dt.days < 0)].copy()
    
    def get_charts_amount_outstanding(self):
        """
        Charts Amount outstanding
        """
        df_opportunity_set_first_date = self.opportunity_set.loc[(self.opportunity_set.ASOF_DATE == self.list_asof_dates[0])].copy()
        
        # Histogram Amount Outstanding
        fig = plt.figure(figsize=(10,5))
        self.histogram_amt_out = fig.add_subplot(1, 2, 1)
        self.histogram_amt_out.hist(df_opportunity_set_first_date.AMOUNTOUTSTANDING/1000000, bins=20, color="darkblue")
        self.histogram_amt_out.set_xlabel("Amount Outstanding ($bn)")
        self.histogram_amt_out.set_ylabel("Number of Securities")
        self.histogram_amt_out.set_title("Histogram Amount Outstanding - 30 June 2022")
        self.histogram_amt_out.grid(True, which='major', linestyle='-', color='lightgray', alpha=0.5)

        # Box-Plot Amount Outstanding
        self.box_plot_amt_out = fig.add_subplot(1, 2, 2)
        self.box_plot_amt_out.boxplot(df_opportunity_set_first_date.AMOUNTOUTSTANDING/1000000, vert=False, labels = [""])
        self.box_plot_amt_out.set_xlabel("Amount Outstanding ($bn)")
        self.box_plot_amt_out.set_title("Box Plot Amount Outstanding - 30 June 2022")
        self.box_plot_amt_out.grid(True, which='major', linestyle='-', color='lightgray', alpha=0.5)
        
        output_file_charts = os.path.join(os.getcwd(), "output", 'Charts_Amount_Outstanding.jpeg')
        plt.savefig(output_file_charts)
        #plt.show()
 
    
    def get_opportunity_set_stats(self):
        """
        This method returns some Summary Statistics of the Investment Universe
        """
        # temp dictionaries to store data
        dict_dur = {}
        dict_ytw = {}
        dict_n_sec = {}
        dict_mv = {}
        dict_amt_out = {}

        # computing various metrics
        for date in self.list_asof_dates:
            temp_df = self.opportunity_set.loc[(self.opportunity_set.ASOF_DATE == date)].copy()
            temp_df["weight"] = temp_df["MARKETVALUE"].div(temp_df["MARKETVALUE"].sum())
            dict_dur[date] = {"MODDURTOWORST": ((temp_df["weight"]*temp_df["MODDURTOWORST"]).sum())}
            dict_ytw[date] = {"YIELDTOWORST": ((temp_df["weight"]*temp_df["YIELDTOWORST"]).sum())}
            dict_n_sec[date] = {"N_SECURITES": temp_df["weight"].count()}
            dict_mv[date] = {"MARKETVALUE": (temp_df["MARKETVALUE"].sum()*1000)}
            dict_amt_out[date] = {"AMOUNTOUTSTANDING": (temp_df["AMOUNTOUTSTANDING"].sum()*1000)}

        # returning the timeseries of values in a single df
        list_dict = [dict_dur,dict_ytw,dict_n_sec,dict_mv,dict_amt_out]
        return pd.concat([pd.DataFrame(i) for i in list_dict]).T

    
    def print_output1(self):
        """
        Method to save the output in a csv file
        """
        output_file_path1 = os.path.join(os.getcwd(), "output", 'Data_Cleaning_output1.csv')
        self.data.to_csv(output_file_path1)
        output_file_path2 = os.path.join(os.getcwd(), "output", 'Opportunity_Set_output2.csv')
        self.opportunity_set.to_csv(output_file_path2)
 
        


class Benchmark(InvestmentUniverse):
    """
    This class is used to track the performace of the Benchmark.
    It's a child class of the InvestmentUniverse class
    """
    def __init__(self):
        super().__init__()
        
        # initializing attributes
        self.eligible_sec_weights = None
        self.sec_returns = None
        self.index_cumulative_returns = None
        self.index_returns = None
        
        # methods to get both benchmark's weights and returns
        self.get_weights()
        self.get_portfolio_returns()
        self.print_output3()
        
    
    def get_weights(self):
        """ 
        Method used to find the constituents' weights.
        This is specific to each Investment strategy.
        In this case, the index is going to be market value weighted.
        """
        # select only securities above the min threshold for amount outstanding
        df_ = self.opportunity_set.loc[self.opportunity_set.AMOUNTOUTSTANDING >= 50000000].copy()
        # Creating a key identifier for each pair ISIN as ASOF_DATE
        df_["temp_identifier"] = (df_.ASOF_DATE.dt.date).astype("str") + "_" + df_.ISIN
        # Temporary dictionary to store results 
        dict_results = {}
        # for each date
        for date_ in self.list_asof_dates:
            df_2 = df_.loc[(df_.ASOF_DATE == date_)].copy()
            # select securities that were issued before the as of date
            df_2.loc[(df_2.ASOF_DATE == date_) & 
                     ((df_2.ISSUEDATE.dt.year <= df_2.ASOF_DATE.dt.year) & 
                     (df_2.ISSUEDATE.dt.month <= df_2.ASOF_DATE.dt.month))].copy()
            # market value weighted
            df_2["Weights_post_rebal"] = (df_2.AMOUNTOUTSTANDING * df_2.PRICEFULL / 100).div((df_2.AMOUNTOUTSTANDING * df_2.PRICEFULL / 100).sum())
            # store results 
            dict_results.update(df_2[["temp_identifier", "Weights_post_rebal"]].set_index("temp_identifier").to_dict()["Weights_post_rebal"])
        # creating a dataframe with the raw results
        df_results = pd.DataFrame([dict_results]).T
        df_results.columns = ["Weights_post_rebal"]
        # table with all the raw data
        self.eligible_sec_weights = pd.merge(df_, df_results, left_on = "temp_identifier", right_index = True, how = "inner")
        # matrix of weights
        self.weights = self.eligible_sec_weights.pivot_table(index = 'ASOF_DATE', columns = 'ISIN', values = 'Weights_post_rebal', aggfunc = 'sum').fillna(0)
        # matrix of returns 
        self.sec_returns = self.eligible_sec_weights.pivot_table(index = 'ASOF_DATE', columns = 'ISIN', values = 'TOTALRETURNMTD', aggfunc = 'sum').fillna(0)

    def get_portfolio_returns(self):
        """
        Method used to compute returns of the portfolio 
        """
        df_w = self.weights.copy()
        df_ret = self.sec_returns.copy()
        # removing rows that are not needed
        df_w = df_w.iloc[:-1,:]
        df_ret = df_ret.iloc[1:,:]
        # realign index
        df_w.index = df_ret.index
        # realign columns
        df_w = df_w[df_ret.columns]
        # compute benchmark returns 
        self.index_cumulative_returns = ((df_w * (df_ret + 1)).sum(1)).cumprod()
        self.index_returns = ((df_w * (df_ret + 1)).sum(1)) - 1
    
    def print_output3(self):
        """
        Method to save the output in a csv file
        """
        output_file_path3 = os.path.join(os.getcwd(), "output", 'Benchmark_Performance_output3.csv')
        df_out = pd.concat([self.index_returns, self.index_cumulative_returns*100], axis=1)
        df_out.columns = ["Benchmark Returns","Benchmark Cumulative Returns"]
        df_out.to_csv(output_file_path3)


class AlphaStrategy(Benchmark):
    """
    This class is used to track the performance of an Alpha generating strategy using the Nelson&Siegel Model.
    It's a child class of the Benchmark class.
    This is still an index strategy, so weights are optimized to achieve the investment objective (see the report for more details).
    This strategy will be implemented in the Portfolio Class
    Weights are the result of an optimization carried out in the OptoEngine class.
    The NelsonSiegelModel class is used to fit the model.
    """

    def __init__(self):
        super().__init__()
        self.model_goodness_fit = self.get_model_fit()
        self.print_output4()
        
    def get_weights(self):
        """ 
        Method used to find the constituents' weights.
        This is specific to each Investment strategy.
        In this case, the vector of weights is the result of an optimization carried out in the OptoEngine class.
        The NelsonSiegelModel class is used to fit the model.
        """
        # assuming that the the previous benchmark is the Parent index of the new strategy
        df_ = self.opportunity_set.loc[self.opportunity_set.AMOUNTOUTSTANDING >= 50000000].copy()
        
        # temporary identifier and dictionaries to store results 
        df_["temp_identifier"] = (df_.ASOF_DATE.dt.date).astype("str") + "_" + df_.ISIN
        dict_weights_results = {}
        self.dict_model_results = {}
        
        # for each date, I select the relevant opportunity set and I compute the % weights
        for date_ in self.list_asof_dates:
            df_2 = df_.loc[(df_.ASOF_DATE == date_)].copy()
            df_2.loc[(df_2.ASOF_DATE == date_) & 
                     ((df_2.ISSUEDATE.dt.year <= df_2.ASOF_DATE.dt.year) & 
                     (df_2.ISSUEDATE.dt.month <= df_2.ASOF_DATE.dt.month))].copy()
            df_2["PCT_MV"] = df_2["MARKETVALUE"].div(df_2["MARKETVALUE"].sum())
            
            # benchmark modified duration and YTW
            bench_mod_duration = df_2["PCT_MV"].dot(df_2["MODDURTOWORST"])
            bench_YTW = df_2["PCT_MV"].dot(df_2["YIELDTOWORST"])
            
            # running the model and getting the overweighted/underweighed bonds
            self.dict_model_results[date_] = NelsonSiegelModel(df_2)
            df_buy = self.dict_model_results[date_].top_buys
            df_sell = self.dict_model_results[date_].top_sells
            
            # lists ISIN overweights/underweights
            buys = (df_buy.loc[(df_buy.CUM_PCT_MV <= 0.25),"ISIN"]).to_list()
            sells = (df_sell.loc[(df_sell.CUM_PCT_MV <= 0.25),"ISIN"]).to_list()

            # running the optimization to find the new weights and sotoring the data in a temporary dictionary
            df_3 = OptoEngine(df_2,buys, sells).data
            dict_weights_results.update(df_3[["temp_identifier", "NEW_PCT_MV"]].set_index("temp_identifier").to_dict()["NEW_PCT_MV"])
        
        # storing results in a dataframe
        df_results = pd.DataFrame([dict_weights_results]).T
        df_results.columns = ["WEIGHTS_ALPHA_INDEX"]
        
        # raw data
        self.eligible_sec_weights = pd.merge(df_, df_results, left_on = "temp_identifier", right_index = True, how = "inner")
        # matrix of weights
        self.weights = self.eligible_sec_weights.pivot_table(index = 'ASOF_DATE', columns = 'ISIN', values = 'WEIGHTS_ALPHA_INDEX', aggfunc = 'sum').fillna(0)
        # matrix of returns
        self.sec_returns = self.eligible_sec_weights.pivot_table(index = 'ASOF_DATE', columns = 'ISIN', values = 'TOTALRETURNMTD', aggfunc = 'sum').fillna(0)

    def get_model_fit(self):
        """ Method returns a table with the goodness of the fit """
        return (pd.DataFrame({date: {"MEAN_SQUARED_ERROR_%": self.dict_model_results[date].mse,
            "ROOT_MEAN_SQUARED_ERROR_%": self.dict_model_results[date].rmse,
            "MEAN_ABSOLUTE_ERROR_%": self.dict_model_results[date].mae,
            "R_SQUARED_%": self.dict_model_results[date].r_squared} for date in self.list_asof_dates})*100).T

    def print_output4(self):
        """ Method to save the output in a csv file """
        output_file_path4 = os.path.join(os.getcwd(), "output", 'Fit_Goodness_output4.csv')
        self.model_goodness_fit.to_csv(output_file_path4)


class NelsonSiegelModel():
    """
    This class is used to fit the NelsonSiegelModel on the bonds in the opportunity set
    This class uses scipy.minimize function
    Input:
    df_: pandas dataframe containing the investment universe as of a particular date
    """
    def __init__(self, df_):
        #initializing the attributes needed for the optimization
        self.raw_data = df_
        self.raw_data["TTM"] = round((self.raw_data.MATURITYDATE - self.raw_data.ASOF_DATE).dt.days / 365, 2)
        self.raw_data.reset_index(drop = True, inplace = True)
        self.raw_data = self.raw_data.sort_values(by="TTM")
        self.yields = self.raw_data["YIELDTOWORST"].values
        self.TTM = self.raw_data['TTM'].values
        self.plot_model_fit = None
        # calling the method to run the actual optimization
        self.get_NelsonSiegelYield()
         
    def compute_NS_Yield(self, params):
        """ Define the spot rate as per the Nelson&Siegel model """
        beta0, beta1, beta2, k = params
        return beta0 + beta1 * ((1-np.exp(- k * self.TTM))/(k * self.TTM)) + beta2 * (((1-np.exp(- k * self.TTM))/( k * self.TTM)) - np.exp(- k * self.TTM))

    def SSE(self, params):
        """ Define the objective function for optimization: minimize SSE """
        return np.sum((self.compute_NS_Yield(params) - self.yields)**2)
    
    def NS_optimization(self, params):
        """Calling the minimization function"""
        return minimize(self.SSE, params)
    
    
    def get_NelsonSiegelYield(self):
        """
        This method runs the optimization and saves the results in various attributes 
        """
        # initial guess for the parameters
        self.params0 = [0.03, -0.1, -0.9, 0.5]
        # actual optimization
        self.res = self.NS_optimization(self.params0)
        # optimized parameters
        self.params_opt = self.res.x
        # predicted yields
        self.predicted_yields = self.compute_NS_Yield(self.params_opt)
        # measures of the goodness of the fit
        self.mse = ((self.predicted_yields - self.yields) ** 2).mean()
        self.rmse = ((self.predicted_yields - self.yields) ** 2).mean() ** 0.5
        self.y_mean = self.yields.mean()
        self.mae = (abs(self.predicted_yields - self.yields)).mean()
        self.ss_tot = np.sum((self.yields - self.y_mean)**2)
        self.ss_res = np.sum((self.yields - self.predicted_yields)**2)
        self.r_squared = 1 - (self.ss_res / self.ss_tot)
        if (self.r_squared < 0.50):
            print("R Squared below 50%")       
        # saving results
        self.raw_data["predicted_yields"] = self.predicted_yields
        self.raw_data["diff_actual_predicted"] = self.raw_data["YIELDTOWORST"] - self.raw_data["predicted_yields"]
        self.raw_data["abs_diff_actual_predicted"] = abs(self.raw_data["YIELDTOWORST"] - self.raw_data["predicted_yields"])
        self.raw_data.loc[(self.raw_data["diff_actual_predicted"] > 0), "BuySell"] = "Buy"
        self.raw_data.loc[(self.raw_data["diff_actual_predicted"] < 0), "BuySell"] = "Sell"
        self.raw_data = self.raw_data.sort_values(by = "abs_diff_actual_predicted", ascending = False)
        self.raw_data["PCT_MV"] = self.raw_data["MARKETVALUE"].div(self.raw_data["MARKETVALUE"].sum())
        # top 25% undervalued bonds (25% in terms of market value weight)
        self.top_buys = self.raw_data.loc[self.raw_data["BuySell"] == "Buy"].copy()
        self.top_buys["CUM_PCT_MV"] = self.top_buys["PCT_MV"].cumsum() 
        self.top_buys.reset_index(drop = True, inplace = True)
        # top 25% overvalued bonds (25% in terms of market value weight)
        self.top_sells = self.raw_data.loc[self.raw_data["BuySell"] == "Sell"].copy()
        self.top_sells["CUM_PCT_MV"] = self.top_sells["PCT_MV"].cumsum() 
        self.top_sells.reset_index(drop = True, inplace = True)

        
class OptoEngine():
    """
    This class runs an optimization to find the optimized weights of the portfolio such that:
    Objective functions:
    Minimize the difference between the Modify Duration To Worst of the strategy and the Parent index
    Minimize the difference between the Yield To Worst of the strategy and the Parent index

    Constraints:
    The top 25% (weight) most overvalued bonds receive a weight equal to zero
    The top 25% (weight) most undervalued bonds are going to be overweighted by at least 25%
    Each constituent is capped at 5%

    Input:
    df_: pandas dataframe containing the investment universe as of a particular date
    buys_: list of strings where each string is the ISIN of one of the bonds that need to be overweighted
    sells_: list of strings where each string is the ISIN of one of the bonds that need to be underweighted
    """
    def __init__(self, df_, buys_, sells_):
        #initializing the attributes needed for the optimization
        self.data = df_.copy()
        self.data.reset_index(drop = True, inplace = True)
        self.buys = buys_
        self.sells = sells_
        self.ytw_benchmark = self.data["PCT_MV"].dot(self.data["MODDURTOWORST"])
        self.dur_benchmark = self.data["PCT_MV"].dot(self.data["YIELDTOWORST"])
        self.w0 = self.data['PCT_MV']
        self.buys_indices = self.data[self.data['ISIN'].isin(self.buys)].index
        self.sells_indices = self.data[self.data['ISIN'].isin(self.sells)].index
        # calling the optimizer
        self.opto_res = self.get_optimized_weights()
        # optimized weights
        self.optimized_weights = self.opto_res.x
        self.data['NEW_PCT_MV'] = self.optimized_weights

        
    def portfolio_optimization(self, w):
        """
        Define the objective function to be minimized: 
        Minimize the absolute % change of YTW and Duration compared with the benchmark (more importance is give to duration)
        Input: np.array of weights to be optimized
        """
        ytw_weighted_avg = np.dot(w[self.buys_indices], self.data.loc[self.buys_indices, 'YIELDTOWORST'])
        dur_weighted_avg = np.dot(w[self.buys_indices], self.data.loc[self.buys_indices, 'MODDURTOWORST'])
        ytw_diff = abs(ytw_weighted_avg - self.ytw_benchmark) / self.ytw_benchmark
        dur_diff = abs(dur_weighted_avg - self.dur_benchmark) / self.dur_benchmark
        objective = 0.25 * ytw_diff + 0.75 * dur_diff
        return objective

    # Define the constraints
    def sum_weights_constraint(self, w):
        """ Weights sum to 1"""
        return 1 - sum(w)

    def buys_constraint(self, w):
        """most undervalued bonds are going to be overweighted by at least 25%"""
        return w[self.buys_indices] - self.data.loc[self.buys_indices, 'PCT_MV'] - (self.data.loc[self.sells_indices, 'PCT_MV'].sum())/len(self.data.loc[self.sells_indices, 'PCT_MV'])

    def sells_constraint(self, w):
        """ Weight of underweighted bonds equal to zero"""
        return w[self.sells_indices]

    def max_weight_constraint(self, w):
        """ Cap at 5% at the security level"""
        return 0.05 - w

    def get_optimized_weights(self):
        """ optimization function """
        return minimize(self.portfolio_optimization, self.w0,
                       constraints=[{'type': 'eq', 'fun': self.sum_weights_constraint},
                                    {'type': 'ineq', 'fun': self.buys_constraint},
                                    {'type': 'eq', 'fun': self.sells_constraint},
                                    {'type': 'ineq', 'fun': self.max_weight_constraint}],
                       bounds=[(0, None) for i in range(len(self.w0))])


class Portfolio(AlphaStrategy):
    """
    This function represents an actual investment portfolio.
    Its investment objective is to replicate as much as possible the strategy outlined in the AlphaStrategy class.
    Input:
    init_ptf_size: integer representing the initial investment in the portfolio, default = 100000000
    """
    def __init__(self, init_ptf_size = 100000000):
        super().__init__()
        #initializing some attributes to store data
        self.init_ptf_size = init_ptf_size
        self.dict_data_Portfolio = {}
        self.dict_data_Alpha_Index = {}
        self.dict_data_Parent_Index = {}
        self.df_raw_data = None
        self.performance_output = None
        # running the replication strategy
        self.run()
        self.print_output5()
        
    # dictionary to store results
    def run(self):
        """
        Method used to run the replication strategy
        """
        # setting some arbitrary amounts for the denomination of each bond
        self.eligible_sec_weights['USD_AMOUNTOUTSTANDING'] = self.eligible_sec_weights['AMOUNTOUTSTANDING'] * 1000
        self.eligible_sec_weights.loc[(self.eligible_sec_weights['USD_AMOUNTOUTSTANDING'] <= 20000000000), 'BOND_DENOMINATION'] = 100
        self.eligible_sec_weights.loc[(self.eligible_sec_weights['USD_AMOUNTOUTSTANDING'] > 20000000000) & (self.eligible_sec_weights['USD_AMOUNTOUTSTANDING'] <= 100000000000) , 'BOND_DENOMINATION'] = 1000
        self.eligible_sec_weights.loc[(self.eligible_sec_weights['USD_AMOUNTOUTSTANDING'] > 100000000000) , 'BOND_DENOMINATION'] = 5000
        self.eligible_sec_weights['MV_ONE_UNIT'] = self.eligible_sec_weights['BOND_DENOMINATION'] * self.eligible_sec_weights['PRICEFULL'] / 100
        
        # temporary variables 
        temp_df_raw_data = []
        temp_ptf_size = self.init_ptf_size
        
        # looping through the various dates
        for date_indicator in range(len(self.list_asof_dates)):    
            temp_df = self.eligible_sec_weights.loc[(self.eligible_sec_weights.ASOF_DATE == self.list_asof_dates[date_indicator])].copy()
            # here I am sourcing the next month's return as the TOTALRETURNMTD refers to the month just concluded
            try:
                temp_df["next_month_identifier"] = (self.list_asof_dates[date_indicator + 1]) + "_" + temp_df["ISIN"]
                temp_df_returns = self.data.loc[(self.data.ASOF_DATE == self.list_asof_dates[date_indicator + 1]), ['temp_identifier','TOTALRETURNMTD']]
                temp_df_returns.columns = ["temp_identifier2", "next_TOTALRETURNMTD"]
                temp_df = pd.merge(temp_df, temp_df_returns, left_on = 'next_month_identifier', right_on = "temp_identifier2", how = "left") 
            # exception handling for the last date    
            except IndexError:
                temp_df["next_TOTALRETURNMTD"] = 0            
            
            # benchmark's stats
            temp_df["BENCH_WEIGHTS"] = (temp_df["MARKETVALUE"]).div(temp_df["MARKETVALUE"].sum()) 
            benchmark_duration = (temp_df['BENCH_WEIGHTS']).dot(temp_df['MODDURTOWORST'])
            benchmark_ytw = (temp_df['BENCH_WEIGHTS']).dot(temp_df['YIELDTOWORST'])
            benchmark_index_return = (temp_df['BENCH_WEIGHTS']).dot((1 + temp_df["next_TOTALRETURNMTD"])) - 1
            
            # alpha (index) strategy's stats
            alpha_index_duration = (temp_df['WEIGHTS_ALPHA_INDEX']).dot(temp_df['MODDURTOWORST'])
            alpha_index_ytw = (temp_df['WEIGHTS_ALPHA_INDEX']).dot(temp_df['YIELDTOWORST'])
            alpha_index_return = (temp_df['WEIGHTS_ALPHA_INDEX']).dot((1 + temp_df["next_TOTALRETURNMTD"])) - 1
            
            # optimal amount (in $) that I should buy for replication purposes
            temp_df['TARGET_MV'] = temp_df['WEIGHTS_ALPHA_INDEX'] * temp_ptf_size
            # actual number of securities bought
            temp_df['QUANTITY'] =  self.get_quantities(temp_df, temp_ptf_size, alpha_index_duration)
            # actual $ position for each sec
            temp_df['POSITION'] = temp_df['QUANTITY'] * temp_df["MV_ONE_UNIT"]
            # total $ amount invested
            amount_invested = temp_df['POSITION'].sum()
            # total $ amount left in cash
            cash = temp_ptf_size - amount_invested
            # % weight invested in bonds
            weight_invested = amount_invested / temp_ptf_size
            # effective % weight in each security (adjusted for cash)
            temp_df['EFFECTIVE_WEIGHT'] = (temp_df['POSITION'].div(temp_df['POSITION'].sum())) * weight_invested
            
            # portfolio's stats
            ptf_modified_duration = (temp_df['EFFECTIVE_WEIGHT']).dot(temp_df['MODDURTOWORST'])
            ptf_ytw = (temp_df['EFFECTIVE_WEIGHT']).dot(temp_df['YIELDTOWORST'])
            ptf_return = (temp_df['EFFECTIVE_WEIGHT']).dot((1 + temp_df["next_TOTALRETURNMTD"])) - 1
            
            # saving results in dictionaries
            self.dict_data_Portfolio[self.list_asof_dates[date_indicator]] = {'Portfolio_Value': temp_ptf_size,
                                              'Amount_Invested': amount_invested,
                                              'Cash_Position': cash,
                                               'Portfolio_YTW': ptf_ytw,
                                               'Portfolio_Duration': ptf_modified_duration,
                                                'Portfolio_Return': ptf_return}
            self.dict_data_Alpha_Index[self.list_asof_dates[date_indicator]] = {
                                               'Alpha_Index_YTW': alpha_index_ytw,
                                               'Alpha_Index_Duration':alpha_index_duration,
                                                'Alpha_Index_Return':alpha_index_return}
            self.dict_data_Parent_Index[self.list_asof_dates[date_indicator]] = {
                                               'Parent_Index_YTW': benchmark_ytw,
                                               'Parent_Index_Duration':benchmark_duration,
                                                'Parent_Index_Return':benchmark_index_return}
            #saving in a temp list
            temp_df_raw_data.append(temp_df)
            # new portfolio market value
            temp_ptf_size = (temp_df['POSITION']).dot((1 + temp_df["next_TOTALRETURNMTD"]))
        
        #saving raw data
        self.df_raw_data = pd.concat(temp_df_raw_data)
        self.format_data_performance()
            
    
    def get_quantities(self, df_, temp_ptf_size_, alpha_index_duration_):
        """
        Method needed to get the actual number of secuities bought to minimize Tracking Error with the Alpha strategy
        while avoiding having a negative cash position
        df_: pandas dataframe containing the investment universe as of a particular date
        temp_ptf_size_: integer representing the $ size of the portfolio
        alpha_index_duration_: float representing the duration of the Alpha Index Strategy that we want to replicate
        """
        # def initial guess for quantity 
        df_['QUANTITY'] =  round(df_['TARGET_MV']/df_['MV_ONE_UNIT'])
        # if cash < 0, then start selling the security with the closest duration to the Alpha Index Startegy's one
        # keep selling until cash > 0
        while ((temp_ptf_size_ - (df_['QUANTITY'] * df_["MV_ONE_UNIT"]).sum()) < 0):
            # don't consider securities with weight equal to zero
            dx = df_.loc[(df_["WEIGHTS_ALPHA_INDEX"] > 0) & (df_["QUANTITY"] > 0)].copy()
            # rank by duration: the more similar to the target duartion of the Alpha strategy, the better
            dx["diff_target_dur"] = (dx["MODDURTOWORST"] - alpha_index_duration_)**2
            dx = dx.sort_values(by = 'diff_target_dur', ascending = True)
            dx.reset_index(drop = True, inplace = True)
            sell_security = str(dx["ISIN"][0])
            df_.loc[(df_["ISIN"] == sell_security), "QUANTITY"] -= 1
        return df_['QUANTITY']
            
    def format_data_performance(self):
        """This method is used to create a table comparing the performances of the three investment strategies"""
        
        # Alpha strategy
        df_alpha = pd.DataFrame(self.dict_data_Alpha_Index).T
        df_alpha['Alpha_Index_Return'] = df_alpha['Alpha_Index_Return'].shift(1, fill_value = 0)
        df_alpha["Alpha_Cumulative_Performance"] = (1 + df_alpha["Alpha_Index_Return"]).cumprod()
        # Portfolio 
        df_portfolio = pd.DataFrame(self.dict_data_Portfolio).T
        df_portfolio['Portfolio_Return'] = df_portfolio['Portfolio_Return'].shift(1, fill_value = 0)
        df_portfolio["Portfolio_Cumulative_Performance"] = (1 + df_portfolio["Portfolio_Return"]).cumprod()
        # Benchmark
        df_parent = pd.DataFrame(self.dict_data_Parent_Index).T
        df_parent['Parent_Index_Return'] = df_parent['Parent_Index_Return'].shift(1, fill_value = 0)
        df_parent["Parent_Cumulative_Performance"] = (1 + df_parent["Parent_Index_Return"]).cumprod()
        # Saving the table in an attribute
        self.performance_output = pd.concat([df_portfolio, df_alpha, df_parent], axis = 1)

    def print_output5(self):
        output_file_path5 = os.path.join(os.getcwd(), "output", 'Portfolio_Performance_output5.csv')
        self.performance_output.to_csv(output_file_path5)
        
