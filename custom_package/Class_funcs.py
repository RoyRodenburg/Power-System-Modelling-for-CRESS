import numpy as np
import matplotlib.pyplot as plt
import pypsa


class CRESS_network:
	"""
	Class to create and analyse different networks, with different solar and battery sizes, and tariff structures.

	Parameters:
		method (str): 'no_grid_charging' or 'grid_charging'
		config (dict): configuration dictionary containing all necessary parameters:
			holiday_arr (arrray [365]): list of holiday days in the year
			peak_hours (array [24]): list of peak hours in a day 
			peak_profile_simplistic (bool): whether to use simplistic peak profile (all weekdays) or realistic (with holidays)
			grid_import_profile (array [8760]): binary array indicating hours when grid charging is allowed
			solar_profile (array [8760]): solar generation profile (per kWp installed)
			kW_solar (float): installed solar capacity in kWp
			battery_efficiency (float): round-trip efficiency of the battery
			BESS_capacity (float): battery capacity in kWh
			finanicials (dict): financial parameters
				wheeling_charge (float): sen/kWh - charge for using the grid to wheel electricity
				electricity_price_grid (float): sen/kWh - base electricity price from the grid
				capacity_charge (float): sen/kW/month
				network_charge (float): sen/kW/month
				off_peak_charge (float): sen/kWh
				peak_charge (float): sen/kWh
				peak_charge_plus (float): sen/kWh - Fictional charge used to ensure that the optimisation prefers solar + battery over grid power during peak hours
				green_electricity_tariff (float): sen/kWh
				PV_cost_per_kWp (float): MYR per kWp
				BESS_cost_per_kWh (float): MYR per kWh
				grid_charging_tariff (float): sen/kWh
				peak_quadratic_cost (float): sen/kWh^2 - quadratic cost component for peak grid power, to penalize high peak power usage
		
		label (str): label for the network instance
	"""

	def __init__(self, method, config, label): 
		self.method = method
		self.config = config
		self.label = label

		self.finanicials = config["finanicials"]
		self.kW_load = 1

	def initialize_network(self):
		"""Initialize the CRESS network based on the specified method and configuration."""
		self.initialize_time_profile()
		self.initialize_financials()
		self.initialize_battery()

		if self.method == 'no_grid_charging':
			self.n = self.create_minimal_CRESS_network()

		if self.method == 'grid_charging':
			self.n = self.create_grid_charging_CRESS_network()



	def initialize_time_profile(self):
		"""Initialize time profile parameters based on config."""
		self.solar_profile = self.config["solar_profile"]
		self.snapshots = range(len(self.solar_profile))
		self.n_days = len(self.solar_profile)//24
		self.grid_import_profile = self.config["grid_import_profile"]

		self.months_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

		self.create_peak_profile()

	def initialize_financials(self):
		"""Initialize financial parameters based on config."""

		fin = self.finanicials

		self.finanicials["yearly_solar_cost"] = (self.finanicials["PV_per_kWp"]*self.config["kW_solar"])
		self.finanicials["yearly_battery_cost"] = (self.finanicials["BESS_per_kWh"]*self.config["BESS_capacity"])
		self.finanicials["base_tariff_off_peak"] = (fin["electricity_price_grid"] 
										+ fin["green_electricity_tariff"] 
										+ fin["off_peak_charge"])
		self.finanicials["base_tariff_peak"] = (fin["electricity_price_grid"] 
								+ fin["green_electricity_tariff"] 
								+ fin["peak_charge"])


	def initialize_battery(self):
		"""Initialize battery parameters based on config."""

		self.config["BESS_power"] = self.config["kW_solar"]-1
		self.config["BESS_max_hours"] = self.config["BESS_capacity"]/(self.config["kW_solar"]-1)

	def create_peak_profile(self): 
		"""Create peak profile based on weekdays and holidays."""

		peak_days = np.zeros(self.n_days)
		peak_profile = np.zeros_like(self.solar_profile)
		if self.config["peak_profile_simplistic"]:
			for i_day in range(self.n_days):
				peak_profile[24*i_day+self.config["peak_hours"]] = 1
				peak_days[i_day] = 1
		for i_day in range(self.n_days):
			if (i_day+2)%7<5 and i_day not in self.config["holiday_arr"]: # +2 because 1st Jan 2025 is a Wednesday
				peak_profile[24*i_day+self.config["peak_hours"]] = 1
				peak_days[i_day] = 1
		print("Number of peak hours in the year:", np.sum(peak_profile))
		self.peak_profile = peak_profile	
		
		self.peak_days = peak_days
	

	def create_minimal_CRESS_network(self):
		""""Create CRESS network without grid charging capability."""

		n = pypsa.Network(snapshots=self.snapshots)
		fin = self.finanicials
		n.add("Bus", "zone")

		n.add("Load", 
			"load", 
			bus="zone", 
			p_set = 1,
			)

		n.add("Generator",
			"solar",
			bus="zone",
			p_nom=self.config["kW_solar"],                   
			p_max_pu=self.solar_profile,
			marginal_cost=0,
			carrier="solar"
		)

		n.add("Generator",
				"grid_power_peak",
				bus="zone",
				p_nom=1,
				committable=True,

			    stand_by_cost=0,
				p_min_pu=1,
				up_time_before=0,
				p_max_pu=self.peak_profile,
				marginal_cost=self.finanicials["base_tariff_peak"]+ fin["peak_charge_plus"],
				min_up_time=3, # must run for at least 3 hours once started, flattens it profile
				marginal_cost_quadratic=self.finanicials["peak_quadratic_cost"])	

		n.add("Generator",
				"grid_power_off_peak",
				bus="zone",
				p_nom=1,
				p_max_pu=np.maximum(0, 1-self.peak_profile-self.solar_profile),
				marginal_cost=self.finanicials["base_tariff_off_peak"])

		n.add("StorageUnit",
				"battery",
				bus="zone",
				carrier="battery",
				p_nom = self.config["BESS_power"],
				efficiency_store=np.sqrt(self.config["battery_efficiency"]),
				efficiency_dispatch=np.sqrt(self.config["battery_efficiency"]),
				max_hours=self.config["BESS_max_hours"],
				marginal_cost=0.0,
				quadratic_cost=0,
				)
		
		# Dummy generator for grid charging scheme
		# to avoid issues with plotting and stats gathering
		# Should not affect optimization results
		n.add("Generator",
			"grid_charging",
			bus="zone",
			p_nom=0,)
		return n


	def create_grid_charging_CRESS_network(self):
		""""Create CRESS network with grid charging capability."""
	
		n = self.create_minimal_CRESS_network()
		n.remove("Generator", "grid_charging")
		n.add("Generator",
			"grid_charging",
			bus="zone",
			p_max_pu=self.config["grid_import_profile"],
			p_nom=self.config["BESS_power"],
			marginal_cost=self.finanicials["grid_charging_tariff"],
		)
		return n
	

	def fully_grid_powered(self, print_bool=True):
		"""
		Assuming a flat base load profile with no solar generation.

		Outputs the annual electricity cost in RM.
		"""
		fin = self.finanicials

		n_hours = len(self.snapshots)
		if self.n_days == 365:
			n_months = 12
		else:
			n_months = 12*(self.n_days/365)
		peak_ratio = np.sum(self.peak_profile)/n_hours
		kW_load = 1

		total_capacity_charge = (fin["capacity_charge"]+fin["network_charge"]) * kW_load* n_months # Sen

		peak_tariff_charge = fin["base_tariff_peak"] * peak_ratio * n_hours  # Sen
		offpeak_tariff_charge = fin["base_tariff_off_peak"] * (1 - peak_ratio) * n_hours  # Sen

		tariff_dict = {
			"total_capacity_charge": 0.01*total_capacity_charge,
			"peak_tariff_charge": 0.01*peak_tariff_charge,
			"offpeak_tariff_charge": 0.01*offpeak_tariff_charge,
			"solar yearly cost": 0,
			"battery yearly cost": 0,
			"wheeling charge": 0,
			"grid_charging_tariff": 0,
		}

		if print_bool:
			for key, value in tariff_dict.items():
				print(f"{key:<25} {value}")
			print("total cost year:      ", sum(tariff_dict.values()))

		return tariff_dict
	
	def monthly_matching_costs(self, optimize=True, print_bool=True):
		"""
		Calculate total annual electricity costs, when using the CRESS monthly matching scheme,
		assuming a flat base load profile with solar generation + batteries and a grid interconnection capacity equal
		to the flat load profile.

		Parameters:
			optimize (bool): Whether to run the optimisation.
			print_bool (bool): Whether to print the breakdown of costs.
		"""

		if optimize:
			self.n.optimize(
				solver_name="highs",
				solver_options={
					"solver": "ipm",
					"presolve": "on",          # default, but make explicit
					"parallel": "on",
					"threads": 8,              # match your CPU cores
					"time_limit": 3600,        # seconds   
				}
			)
		CRESS_production = np.sum(self.n.generators_t.p["solar"])+np.sum(self.n.storage_units_t.p["battery"])

		print("CRESS kWh per year", CRESS_production)

		fin = self.config["finanicials"]
		peak_hours_sum = np.sum(self.peak_profile)
		n_hours = len(self.snapshots)
		if self.n_days == 365:
			n_months = 12
		else:
			n_months = 12*(self.n_days/365)
		kW_load = 1


		if CRESS_production <= peak_hours_sum:
			peak_consumption = peak_hours_sum - CRESS_production
			off_peak_consumption = n_hours - peak_hours_sum
		else:
			peak_consumption = 0
			off_peak_consumption = n_hours - CRESS_production 


		total_capacity_charge = (fin["capacity_charge"]+fin["network_charge"]) * kW_load* n_months # Sen

		peak_tariff_charge = fin["base_tariff_peak"] * peak_consumption  # Sen
		offpeak_tariff_charge = fin["base_tariff_off_peak"] * off_peak_consumption  # Sen
		tariff_dict = {
			"total_capacity_charge": 0.01*total_capacity_charge,
			"peak_tariff_charge": 0.01*peak_tariff_charge,
			"offpeak_tariff_charge": 0.01*offpeak_tariff_charge,
			"solar yearly cost": fin["yearly_solar_cost"],
			"battery yearly cost": fin["yearly_battery_cost"],
			"wheeling charge": fin["wheeling_charge"] * CRESS_production * 0.01,
			"grid_charging_tariff": 0,
		}

		if print_bool:
			for key, value in tariff_dict.items():
				print(f"{key:<25} {value}")
			print("total cost year:      ", sum(tariff_dict.values()))

		self.finanicials["total_cost_year"] = sum(tariff_dict.values())
		return tariff_dict


	def calculate_total_capacity_charge(self):
		"""
		Calculate total capacity charge based on the maximum grid demand in each month.
		Updates the financials dictionary with the total capacity charge and monthly max grid demand.
		"""
		months_start_arr = np.cumsum([0]+self.months_arr[:-1])
		self.daily_max_grid_demand = [max(self.n.generators_t.p["grid_power_peak"][i*24:(i+1)*24]) for i in range(self.n_days)]
		max_production_arr = []
		for i in range(12):
			if months_start_arr[i] >= self.n_days:
				break
			end_date = min(months_start_arr[i]+self.months_arr[i], self.n_days)
			pow_max = max(self.daily_max_grid_demand[months_start_arr[i]:end_date])
			max_production_arr.append(pow_max)

		fin = self.finanicials
		self.finanicials["total_capacity_charge_year"] = 0.01*self.kW_load * (fin["capacity_charge"] + fin["network_charge"]) *np.sum(max_production_arr)
		self.finanicials["max_grid_demand_month"] = max_production_arr


	def hourly_matching_costs(self, print_bool=True):
		"""
		Compute total annual electricity costs, when using the CRESS hourly matching scheme,
		assuming a flat base load profile with solar generation.

		Outputs the annual electricity cost in RM/year.

		Parameters:
			print_bool (bool): Whether to print the breakdown of costs.
		"""

		if self.n is None:
			print("Network not initialized!")
			return None

		self.calculate_total_capacity_charge()
		fin = self.finanicials
		wheeling_charge_year = 0.01*fin["wheeling_charge"] * (np.sum(self.n.generators_t.p["solar"])+np.sum(self.n.storage_units_t.p["battery"]))


		tariff_dict = {
			"total_capacity_charge": fin["total_capacity_charge_year"],
			"peak_tariff_charge": 0.01*np.sum(self.n.generators_t.p["grid_power_peak"])* fin["base_tariff_peak"],
			"offpeak_tariff_charge": 0.01*np.sum(self.n.generators_t.p["grid_power_off_peak"])* fin["base_tariff_off_peak"],
			"solar yearly cost": fin["yearly_solar_cost"],
			"battery yearly cost": fin["yearly_battery_cost"],
			"wheeling charge": wheeling_charge_year,
			"grid_charging_tariff": 0.01*np.sum(self.n.generators_t.p["grid_charging"])* fin["grid_charging_tariff"],
		}

		self.finanicials["total_cost_year"] = sum(tariff_dict.values())

		if print_bool:
			for key, value in tariff_dict.items():
				print(f"{key:<25} {value}")
			print("total cost year:      ", self.finanicials["total_cost_year"])
		

		return tariff_dict


	def gather_statistics(self):
		"""
		Helper function to gather useful statistics from the network after optimisation.
		Creates a dictionary of statistics.
		"""

		stats = {}
		stats["total_solar_production"] = np.sum(self.n.generators_t.p["solar"])
		stats["total_grid_import_off_peak"] = self.n.generators_t.p["grid_power_off_peak"]
		stats["total_grid_import_peak"] = np.sum(self.n.generators_t.p["grid_power_peak"])
		stats["total_grid_charging"] = np.sum(self.n.generators_t.p["grid_charging"])

		stats["solar_curtailment"] = np.sum(self.solar_profile*self.config["kW_solar"] - self.n.generators_t.p["solar"])

		arr = self.n.storage_units_t.p["battery"]
		stats["total_BESS_discharge"] = np.sum(arr[arr>0])
		stats["BESS_efficiency_losses"] = np.sum(-arr[arr<0]-arr[arr>0])
		if self.config["BESS_capacity"] > 0:
			stats["BESS_utilization_factor"] = 1/np.sqrt(self.config["battery_efficiency"])*stats["total_BESS_discharge"]/(self.config["BESS_capacity"]*self.n_days)
		else:
			stats["BESS_utilization_factor"] = 0

		self.stats = stats
		return stats


	def plot_average_day(self, battery_SOC=True):
		"""
		Create plot for the average day. Shows power profile and battery SOC if specified.
		Parameters:
			battery_SOC (bool): Whether to plot the battery state of charge in a separate subplot.
		"""
		def create_daily_avg(str_label):
			if str_label == "battery":
				gen_arr = np.array(self.n.storage_units_t.p["battery"])
			else:
				gen_arr = np.array(self.n.generators_t.p[str_label])
			return np.mean(gen_arr.reshape(self.n_days,24), axis=0)
		
		fig, ax = plt.subplots(2, 1, figsize=(12, 8 if battery_SOC else 12), sharex=battery_SOC)
		
		arr_label = ["solar", "grid_power_off_peak", "grid_power_peak", "grid_charging", "battery"]
		for str_label in arr_label:
			arr = create_daily_avg(str_label)
			ax[0].step(np.arange(25), np.hstack((arr, arr[-1])), where="post", label=str_label)


		ax[0].set_title("average day of: "+ self.label)
		ax[0].axvline(np.min(self.config["peak_hours"]), linestyle='--', label = "peak hours")
		ax[0].axvline(np.max(self.config["peak_hours"])+1, linestyle='--')
		ax[0].legend()
		ax[0].grid()
		ax[0].set_ylabel("Power (kW)")
	
		if battery_SOC:
			battery_arr = np.array(self.n.storage_units_t.state_of_charge["battery"])
			arr = np.mean(battery_arr.reshape(self.n_days,24), axis=0)
			arr = np.insert(arr, 0, arr[-1])
			ax[1].plot(arr, label="battery")

			ax[1].axvline(np.min(self.config["peak_hours"]), linestyle='--', label = "peak hours")
			ax[1].set_ylabel("Battery Charge (kWh)")
			ax[1].axvline(np.max(self.config["peak_hours"])+1, linestyle='--')
			ax[1].legend()
			ax[1].grid()
			plt.xlabel("Hour of the day")
		else:
			plt.tight_layout()
			ax[0].set_xlabel("Hour of the day")


		return ax, fig


	def plot_worst_day(self, battery_SOC=True):
		"""
		Create a plot for the worst day in terms of peak grid power usage. Shows power profile and battery SOC if specified.
		Parameters:
			battery_SOC (bool): Whether to plot the battery state of charge in a separate subplot.
		"""
		def create_daily_worst(str_label):
			if str_label == "battery":
				gen_arr = np.array(self.n.storage_units_t.p["battery"])
			else:
				gen_arr = np.array(self.n.generators_t.p[str_label])
			return gen_arr[worst_day*24:(worst_day+1)*24]
		


		fig, ax = plt.subplots(2, 1, figsize=(12, 8 if battery_SOC else 12), sharex=battery_SOC)

		peak_arr = np.array(self.n.generators_t.p["grid_power_peak"])
		if np.sum(peak_arr) > 0:
			worst_day = np.argmax(peak_arr.reshape(self.n_days,24).sum(axis=1))
		else:
			grid_charging_arr = np.array(self.n.generators_t.p["grid_charging"])
			worst_day = np.argmax(grid_charging_arr.reshape(self.n_days,24).sum(axis=1))

		print("Worst day index:", worst_day, np.sum(peak_arr))

		arr_label = ["solar", "grid_power_off_peak", "grid_power_peak", "grid_charging", "battery"]
		for str_label in arr_label:
			arr = create_daily_worst(str_label)
			ax[0].step(np.arange(25), np.hstack((arr, arr[-1])), where="post", label=str_label)


		ax[0].set_title("worst day of: "+ self.label)
		ax[0].axvline(np.min(self.config["peak_hours"]), linestyle='--', label = "peak hours")
		ax[0].axvline(np.max(self.config["peak_hours"])+1, linestyle='--')
		ax[0].legend()
		ax[0].grid()
		ax[0].set_ylabel("Power (kW)")

		if battery_SOC:
			battery_arr = np.array(self.n.storage_units_t.state_of_charge["battery"])
			arr = battery_arr[worst_day*24:(worst_day+1)*24]
			arr = np.insert(arr, 0, arr[-1])
			ax[1].plot(arr, label="battery")

			ax[1].axvline(np.min(self.config["peak_hours"]), linestyle='--', label = "peak hours")
			ax[1].set_ylabel("Battery Charge (kWh)")
			ax[1].axvline(np.max(self.config["peak_hours"])+1, linestyle='--')
			ax[1].legend()
			ax[1].grid()
			plt.xlabel("Hour of the day")
		else:
			plt.tight_layout()
			ax[0].set_xlabel("Hour of the day")

		return ax, fig
