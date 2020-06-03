import pandas as pd
import numpy as np
import datetime as dt
from util_funcs import *
import time
from scipy.optimize import minimize

PATH_TO_INPUT_FILES = './data/'

### loading data ###
df_pop = pd.read_csv(PATH_TO_INPUT_FILES+'italy_reg_pop_DEF.csv')
df_covid = pd.read_csv(PATH_TO_INPUT_FILES+'italy_cases_recovered_deaths_DEF.csv', parse_dates=['date'])
df_flows = pd.read_csv(PATH_TO_INPUT_FILES+'italy_movement_24_feb_22_apr_DEF.csv', parse_dates=['date'])
df_lockdown = pd.read_csv(PATH_TO_INPUT_FILES+'italy_lockdown_data.csv', parse_dates=['date'])

### initialization: ###
set_all_regions = set(df_pop['reg_name'])
start_date = dt.datetime(2020, 2, 24, 0, 0, 0)

### all data until 2020-04-22 ###
df_covid = df_covid[df_covid['date'] <= max(df_flows['date'])]
df_lockdown = df_lockdown[df_lockdown['date'] <= max(df_flows['date'])]

### add susceptibles ###
df_all = add_susceptibles_to_df(df_covid, df_pop)

### aggregating covid data ###
df_aggregated = aggregate_SIR_data(df_all, 7)

### aggregating mov. flow by days ###
df_flows_sub = df_flows[['date', 'start_reg_name', 'end_reg_name', 'flow']]
df_flows_agg = df_flows_sub[df_flows_sub.duplicated(['date', 'start_reg_name', 'end_reg_name'], keep=False)].groupby(['date', 'start_reg_name', 'end_reg_name']).apply(lambda x: np.sum(x['flow'])).reset_index()
df_flows_agg.columns = df_flows_sub.columns

### lockdown measures ###
map_Indicator_to_N = {
	'1': 3,
	'2': 3,
	'3': 2,
	'4': 4,
	'5': 2,
	'6': 3,
	'7': 2,
	'8': 4
}
w = (1 / 8) * sum([1 / (N + 1) for N in map_Indicator_to_N.values()])

def compute_I(C, C_num, G=0):
	if C_num != '8':
		I = C * (1 - w) / map_Indicator_to_N[C_num] + w * G
	else:
		# this is for indicator 8, that has no flag 1/0
		I = C / map_Indicator_to_N[C_num]
	return I

def add_I_to_df(df_lockdown):
	df_with_I = df_lockdown.copy()

	for C_num in map_Indicator_to_N.keys():
		if C_num != '8':
			df_with_I['I_' + C_num] = compute_I(df_with_I['C' + C_num], C_num, df_with_I['C' + C_num + '_Flag'])
		else:
			df_with_I['I_' + C_num] = compute_I(df_with_I['C' + C_num], C_num)

	return df_with_I.fillna(0)  # " [...] an absence of data corresponds to a sub-index of zero."

df_Ind = add_I_to_df(df_lockdown)[['date', 'I_1', 'I_2', 'I_3', 'I_4', 'I_5', 'I_6', 'I_7', 'I_8']]



###### MODEL WITH p_r varying #######

start_time = time.time()

class Learner(object):
	def __init__(self, region, loss, start_date):
		self.region = region
		self.loss = loss
		self.start_date = start_date

	### functions to load data: ###
	def load_positives(self, region):
		df = df_covid
		region_df = df[df['reg_name'] == region]
		return region_df[region_df['date'] >= start_date]['positives'].reset_index(drop=True)

	def load_delta_positives(self, region):
		df = df_covid
		region_df = df[df['reg_name'] == region]
		return region_df[region_df['date'] >= start_date]['delta_positives'].reset_index(drop=True)

	def load_removed(self, region):
		df = df_covid
		region_df = df[df['reg_name'] == region]
		return region_df[region_df['date'] >= start_date]['removed'].reset_index(drop=True)

	def load_delta_removed(self, region):
		df = df_covid
		region_df = df[df['reg_name'] == region]
		return region_df[region_df['date'] >= start_date]['delta_removed'].reset_index(drop=True)

	def load_pop(self, region):
		df = df_pop
		region_df = df[df['reg_name'] == region]
		return int(region_df['pop'])  ## pop is fixed

	def load_index(self, list_index):
		df = df_Ind
		return df[df['date'] >= start_date][list_index].reset_index(drop=True)

	def train(self):
		### loading data: ###
		removed = self.load_removed(self.region)
		# print('removed, ', removed.shape)
		delta_removed = self.load_delta_removed(self.region)
		# print('delta_removed, ', delta_removed.shape)
		positives = self.load_positives(self.region)
		# print('positives, ', positives.shape)
		delta_positives = self.load_delta_positives(self.region)
		# print('delta_positives, ', delta_positives.shape)
		# susceptibles = positives - removed
		# print('susceptibles, ', susceptibles.shape)

		population = self.load_pop(self.region)
		# print('population, ', population)

		restrictions = self.load_index(['I_1', 'I_2', 'I_3', 'I_4', 'I_5', 'I_6', 'I_7', 'I_8'])
		# print('restrictions, ', restrictions.shape)

		### initial guess for the params: ###
		start_params = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 1, 0, 1, 0]

		### minimizing the loss function: ###
		optimal = minimize(loss, start_params,
						   args=(positives, removed, delta_positives, delta_removed,
								 population, restrictions, self.region),
						   method='BFGS')
		# print(optimal)
		alpha_b_1, alpha_b_2, alpha_b_3, alpha_b_4, alpha_b_5, alpha_b_6, c_b, alpha_o, c_o, alpha_p, c_p = optimal.x

		return alpha_b_1, alpha_b_2, alpha_b_3, alpha_b_4, alpha_b_5, alpha_b_6, c_b, alpha_o, c_o, alpha_p, c_p


### defining the loss function:
def loss(point, positives, removed, delta_positives, delta_removed,
		 population, restrictions, region):
	### parameters: ###
	alpha_b_1, alpha_b_2, alpha_b_3, alpha_b_4, alpha_b_5, alpha_b_6, c_b, alpha_o, c_o, alpha_p, c_p = point

	### data: ###
	I = positives
	R = removed
	N = population
	S = N - (I + R)

	### gamma, beta, o, p ###
	gamma = 1 / 3
	beta = alpha_b_1 * restrictions['I_1'] + alpha_b_2 * restrictions['I_2'] + alpha_b_3 * restrictions[
		'I_3'] + alpha_b_4 * restrictions['I_4'] + alpha_b_5 * restrictions['I_5'] + alpha_b_6 * restrictions[
			   'I_6'] + c_b
	o = alpha_o * restrictions['I_8'] + c_o
	p = 1 / (1 + np.exp(-(c_p + alpha_p * restrictions['I_7'])))

	### pos from out: ###
	I_out = compute_positives_from_out(region, p)

	### SIR equations: ###
	dSdt = -beta * I * S / N
	dIdt = beta * ((I + I_out + o) * S) / N - gamma * I / N
	dRdt = gamma * I / N

	### scores to minimize: ###
	l1 = np.sqrt(np.mean((dIdt - delta_positives) ** 2))
	l2 = np.sqrt(np.mean((dRdt - delta_removed) ** 2))
	alpha = 0.5

	return alpha * l1 + (1 - alpha) * l2


def compute_positives_from_out(region, p):
	I_out = 0
	for c_region in set_all_regions:
		if c_region != region:
			## L
			c_df = df_flows_agg[(df_flows_agg['end_reg_name'] == region) &
								(df_flows_agg['start_reg_name'] == c_region) &
								(df_flows_agg['date'] >= start_date)][['date', 'flow']]
			df_with_inflow = c_df.copy()
			for c_date in set(df_flows_agg[df_flows_agg['date'] >= start_date]['date']):
				if c_date not in set(c_df['date']) or df_with_inflow.empty:
					df_with_inflow = df_with_inflow.append({'date': c_date, 'flow': 0}, ignore_index=True)

			df_with_inflow = df_with_inflow.sort_values(['date']).reset_index(drop=True)
			L = df_with_inflow['flow']

			## I
			I = df_covid[(df_covid['reg_name'] == c_region) &
						 (df_covid['date'] >= start_date)]['positives'].reset_index(drop=True)

			## N
			N = int(df_pop[df_pop['reg_name'] == c_region]['pop'])

			I_out += p * L * I / N

	return I_out


######


### all the regions:
map__region_to_params = {}
for region in set_all_regions:
	learner = Learner(region, loss, start_date)
	est_params = learner.train()
	map__region_to_params[region] = est_params
	print('%s : %s' %(region, est_params))

### only 1 region:
'''
learner = Learner('TOSCANA', loss, start_date)
est_params = learner.train()
'''

print()
print("runtime: %.2f seconds" % (time.time() - start_time))