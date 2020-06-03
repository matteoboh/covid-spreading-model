import datetime as dt
import numpy as np



################################################################################
############################# MOVEMENT FLOWS ###################################
################################################################################

def extract_flow_in_t(df, name_of_starting_region, name_of_ending_region,
					  time_t=dt.datetime(2020, 2, 24, 0, 0, 0),
					  type_of_region = 'regione'):
	""" Function to extract the flow from r' to r at time t.

	:param df: DataFrame
		for each date, contains information of movement flow: names of starting region and ending region, and the value of flow between them.
	:param name_of_starting_region: str
		name of the region where the flow starts.
	:param name_of_ending_region: str
		name of the region where the flow ends.
	:param time_t: datetime
		the exact date for which one wants to extract the value of flow.
	:param type_of_region: str
		the spatial aggregation. For Italy, one of ['regione', 'provincia'].
	:return: int
		the total value of flow in the period.
	"""
	if type_of_region == 'regione':
		starting_region_column = 'start_reg_name'
		ending_region_column = 'end_reg_name'
	else:
		starting_region_column = 'start_prov_name'
		ending_region_column = 'end_prov_name'

	try:
		flow = int(np.sum(df[(df[starting_region_column] == name_of_starting_region) &
							 (df[ending_region_column] == name_of_ending_region) &
							 (df['date'] == time_t)]['flow']))
	except:
		# this happens if there are no entries in the df with data on flow between the two regions
		# (i.e. the flow between them is 0)
		flow = 0

	return flow


def extract_flow_in_time_interval(df, name_of_starting_region, name_of_ending_region,
								  t_0 = dt.datetime(2020, 2, 24, 0, 0, 0), t_1 = dt.datetime(2020, 2, 25, 0, 0, 0),
								  type_of_region = 'regione'):
	""" Function to extract the flow from r' to r between t_0 and t_1.

	:param df: DataFrame
		for each date, contains information of movement flow: names of starting region and ending region, and the value of flow between them.
	:param name_of_starting_region: str
		name of the region where the flow starts.
	:param name_of_ending_region: str
		name of the region where the flow ends.
	:param t_0: datetime object
		the starting date of the period for which one wants to aggregate the flow.
	:param t_1: datetime object
		the ending date of the period for which one wants to aggregate the flow.
	:param type_of_region: str
		the spatial aggregation. For Italy, one of ['regione', 'provincia'].
	:return: int
		the total value of flow in the period.
	"""
	if type_of_region == 'regione':
		starting_region_column = 'start_reg_name'
		ending_region_column = 'end_reg_name'
	else:
		starting_region_column = 'start_prov_name'
		ending_region_column = 'end_prov_name'

	try:
		flow = int(np.sum(df[(df[starting_region_column] == name_of_starting_region) &
							 (df[ending_region_column] == name_of_ending_region) &
							 (df['date'] >= t_0) &
							 (df['date'] < t_1)]['flow']))
	except:
		flow = 0

	return flow


def extract_inflow_in_time_interval(df, name_of_region,
									t_0 = dt.datetime(2020, 2, 24, 0, 0, 0), t_1 = dt.datetime(2020, 2, 25, 0, 0, 0),
									type_of_region = 'regione'):
	""" Function to extract the total INflow to region r from all the others between t_0 and t_1.

	:param df: DataFrame
		for each date, contains information of movement flow: names of starting region and ending region, and the value of flow between them.
	:param name_of_region: str
		name of the region for which one wants the total inflow.
	:param t_0: datetime object
		the starting date of the period for which one wants to aggregate the inflow.
	:param t_1: datetime object
		the ending date of the period for which one wants to aggregate the inflow.
	:param type_of_region: str
		the spatial aggregation. For Italy, one of ['regione', 'provincia'].
	:return: int
		the total value of inflow in the period.
	"""
	if type_of_region == 'regione':
		starting_region_column = 'start_reg_name'
		ending_region_column = 'end_reg_name'
	else:
		starting_region_column = 'start_prov_name'
		ending_region_column = 'end_prov_name'

	try:
		flow = int(np.sum(df[(df[starting_region_column] != name_of_region) &
							 (df[ending_region_column] == name_of_region) &
							 (df['date'] >= t_0) &
							 (df['date'] < t_1)]['flow']))
	except:
		flow = 0

	return flow


def extract_outflow_in_time_interval(df, name_of_region,
									t_0=dt.datetime(2020, 2, 24, 0, 0, 0), t_1=dt.datetime(2020, 2, 25, 0, 0, 0),
									type_of_region='regione'):
	""" Function to extract the total OUTflow from region r to all the others between t_0 and t_1.

	:param df: DataFrame
		for each date, contains information of movement flow: names of starting region and ending region, and the value of flow between them.
	:param name_of_region: str
		name of the region for which one wants the total outflow.
	:param t_0: datetime object
		the starting date of the period for which one wants to aggregate the outflow.
	:param t_1: datetime object
		the ending date of the period for which one wants to aggregate the outflow.
	:param type_of_region: str
		the spatial aggregation. For Italy, one of ['regione', 'provincia'].
	:return: int
		the total value of outflow in the period.
	"""
	if type_of_region == 'regione':
		starting_region_column = 'start_reg_name'
		ending_region_column = 'end_reg_name'
	else:
		starting_region_column = 'start_prov_name'
		ending_region_column = 'end_prov_name'

	try:
		flow = int(np.sum(df[(df[starting_region_column] == name_of_region) &
							 (df[ending_region_column] != name_of_region) &
							 (df['date'] >= t_0) &
							 (df['date'] < t_1)]['flow']))
	except:
		flow = 0

	return flow


################################################################################
############################### COVID DATA #####################################
################################################################################


#### dealing with N and S ####

def add_pop_to_df(df_covid, df_pop):
	df_with_N = df_covid.groupby(['reg_name']).apply(_add_N, df_pop)
	df_with_N = df_with_N[['date', 'reg_name', 'population'] + list(df_covid.columns[2:])]

	return df_with_N


def _add_N(df, df_pop):
	population = int(df_pop[df_pop['reg_name'] == df['reg_name'].iloc[0]]['pop'])
	array_N = np.repeat(population, df.shape[0])  # N_r is fixed, it does not change with t
	df['population'] = array_N

	return df


def add_susceptibles_to_df(df_covid, df_pop):
	if 'population' not in df_covid.columns:
		df_covid = add_pop_to_df(df_covid, df_pop).copy()

	df_with_S = df_covid.groupby(['reg_name']).apply(_add_S)
	df_with_S = df_with_S[
		['date', 'reg_name', 'population', 'susceptibles', 'delta_susceptibles'] + list(df_covid.columns[3:])]

	return df_with_S


def _add_S(df):
	df['susceptibles'] = df['population'] - (df['positives'] + df['removed'])

	array_S = np.array(df['susceptibles'])
	array_delta_S = np.insert(np.diff(array_S), 0, array_S[0])
	df['delta_susceptibles'] = array_delta_S

	return df


#### data aggregation ####

def aggregate_SIR_data(df, num_of_days_per_group=7):
	"""Aggregates S, I, R, N data by time.

	:param df: DataFrame
		contains columns with daily data on N, S, delta_S, I, delta_I, R, delta_R for each region.
	:param num_of_days_per_group: int
		the number of days one wants to aggregate. Default is 7, for weekly aggregation.
	:return: DataFrame
		a new DataFrame with exactly the same structure as the input one, but with aggregated data:
		each num_of_days_per_group rows in df is aggregated into 1 row, where the values for delta_S, delta_I, delta_R
		are simply the sum of the original values, and S(t), I(t), R(t) are obtained as S=S(first day in group)+delta_S.
		Notes: (1) the date referring to this new aggregated row is the date of the first day in the group,
		(2) N is fixed for each region and does not depend on time.
	"""
	df_aggr = df.groupby(['reg_name']).apply(_aggregate, num_of_days_per_group).reset_index()
	df_aggr = df_aggr[list(df.columns)]

	return df_aggr


def _aggregate(df, num_of_days_per_group):

	# 1. creating the labels for the aggregation:
	# these are stored in a temporary new column ('group_labels') of type [0,0,0,1,1,1,2,2,2,...,k,k,k] if the num_of_days_per_group = 3
	num_of_entries = df.shape[0]
	days = np.arange(0, num_of_entries)
	group_labels = np.repeat(days, num_of_days_per_group)[0:num_of_entries]
	df['group_labels'] = group_labels

	# 2. aggregating the deltas (delta_susceptibles, delta_positives, delta_removed)
	df_aggr = df.groupby(['group_labels']).agg(
		{'delta_susceptibles': 'sum', 'delta_positives': 'sum', 'delta_removed': 'sum'}).reset_index()

	# 3. aggregating population, susceptibles, positives and removed.
	# The value of the last day in each aggregated group is the value we want for that group:
	# (it is exactly the same as adding the delta to the value of the first day in each group)
	last_indexes = [group_indexes[len(group_indexes) - 1] for group_indexes in
					df.groupby(['group_labels']).groups.values()]
	df_aggr['susceptibles'] = np.array(df.loc[last_indexes, 'susceptibles'])
	df_aggr['positives'] = np.array(df.loc[last_indexes, 'positives'])
	df_aggr['removed'] = np.array(df.loc[last_indexes, 'removed'])
	df_aggr['population'] = np.array(df.loc[last_indexes, 'population'])

	# the date of each aggregated group will be the date of the first day in each group:
	first_indexes = [group_indexes[0] for group_indexes in df.groupby(['group_labels']).groups.values()]
	df_aggr['date'] = np.array(df.loc[first_indexes, 'date'])

	return df_aggr