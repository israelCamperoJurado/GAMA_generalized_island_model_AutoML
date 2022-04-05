# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 14:54:11 2021

@author: 20210595
"""

upperBound = dict()

upperBound['SimpleImputer'] = 30
upperBound['SimpleImputer_strategy'] = 3 
upperBound['SimpleImputer_fill_value'] = 0 # max value in the serie
upperBound['AutoReg'] = 100
upperBound['AutoReg_lags'] = 10
upperBound['AutoReg_trend'] = 3
upperBound['ARIMA'] = 100
upperBound['ARIMA_order_p'] = 10
upperBound['ARIMA_order_d'] = 10
upperBound['ARIMA_order_q'] = 10
upperBound['ARIMA_trend'] = 3
upperBound['ARIMA_enforce_stationarity'] = 1
upperBound['ARIMA_enforce_invertibility'] = 1
upperBound['ARIMA_concentrate_scale'] = 1
upperBound['SARIMAX'] = 100
upperBound['SARIMAX_order_p'] = 10
upperBound['SARIMAX_order_d'] = 10
upperBound['SARIMAX_order_q'] = 10
upperBound['SARIMAX_trend'] = 3
upperBound['SARIMAX_enforce_stationarity'] = 1
upperBound['SARIMAX_enforce_invertibility'] = 1
upperBound['ExponentialSmoothing'] = 100
upperBound['ExponentialSmoothing_trend'] = 3
upperBound['ExponentialSmoothing_damped_trend'] = 1
upperBound['ExponentialSmoothing_seasonal'] = 365 #Given in day per year 
upperBound['ExponentialSmoothing_initialization_method'] = 2 # It was include ‘legacy-heuristic’
upperBound['ExponentialSmoothing_initial_level'] = 1
upperBound['ExponentialSmoothing_initial_trend'] = 1
upperBound['Holt'] = 100
upperBound['Holt_exponential'] = 1
upperBound['Holt_damped_trend'] = 1
upperBound['Holt_initialization_method'] = 2 # It was include ‘legacy-heuristic’
upperBound['Holt_initial_level'] = 1
upperBound['Holt_initial_trend'] = 1
upperBound['SimpleExpSmoothing'] = 100
upperBound['SimpleExpSmoothing_initialization_method'] = 2 # It was include ‘legacy-heuristic’
upperBound['SimpleExpSmoothing_initial_level'] = 10
upperBound['ETSModel'] = 100
upperBound['ETSModel_error'] = 1
upperBound['ETSModel_trend'] = 1
upperBound['ETSModel_damped_trend'] = 1
upperBound['ETSModel_seasonal'] = 1
upperBound['ETSModel_seasonal_periods'] = 365 #Given in day per year 
upperBound['ETSModel_initialization_method'] = 2
upperBound['ETSModel_initial_level'] = 1
upperBound['ETSModel_initial_trend'] = 1
upperBound['DynamicFactor'] = 100
upperBound['DynamicFactor_k_factors'] = 10
upperBound['DynamicFactor_factor_order'] = 10
upperBound['DynamicFactor_error_cov_type'] = 2
upperBound['DynamicFactorMQ'] = 100
upperBound['DynamicFactorMQ_k_endog_monthly'] = 10
upperBound['DynamicFactorMQ_factors'] = 10
upperBound['DynamicFactorMQ_factor_orders'] = 10
upperBound['DynamicFactorMQ_factor_multiplicities'] = 10
upperBound['DynamicFactorMQ_idiosyncratic_ar1'] = 1
upperBound['DynamicFactorMQ_standardize'] = 1
upperBound['DynamicFactorMQ_init_t0'] = 1
upperBound['VAR'] = 100
upperBound['VARMAX'] = 100
upperBound['VARMAX_order_p'] = 10
upperBound['VARMAX_order_q'] = 10
upperBound['VARMAX_trend'] = 3
upperBound['VARMAX_error_cov_type'] = 1
upperBound['VARMAX_measurement_error'] = 1
upperBound['VARMAX_enforce_stationarity'] = 1
upperBound['VARMAX_enforce_invertibility'] = 1
upperBound['SVAR'] = 100
upperBound['SVAR_svar_typestr'] = 2
upperBound['VECM'] = 100
upperBound['VECM_k_ar_diff'] = 10
upperBound['VECM_coint_rank'] = 10
upperBound['VECM_deterministic'] = 4
upperBound['UnobservedComponents'] = 100
upperBound['UnobservedComponents_level'] = 1
upperBound['UnobservedComponents_trend'] = 1
upperBound['UnobservedComponents_seasonal'] = 50
upperBound['UnobservedComponents_cycle'] = 1
upperBound['UnobservedComponents_autoregressive'] = 10
upperBound['UnobservedComponents_irregular'] = 1
upperBound['UnobservedComponents_stochastic_level'] = 1
upperBound['UnobservedComponents_stochastic_trend'] = 1
upperBound['UnobservedComponents_stochastic_seasonal'] = 1
upperBound['UnobservedComponents_stochastic_cycle'] =1 
upperBound['UnobservedComponents_damped_cycle'] = 1
upperBound['UnobservedComponents_use_exact_diffuse'] = 1

NewUpperBound = [i for i in upperBound.values()]

NewlowerBound = [0]*len(upperBound)