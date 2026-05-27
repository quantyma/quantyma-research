import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_atmospheric_data_amazon(years=5, start="2021-01-01", g_max_dry=1000, noise_std=300, seed=42):
    """
    Generates synthetic atmospheric and solar irradiance data for the Amazon region (Manaus).
    Calibrated for strict block transitions: wet (Dec-May) and dry (Jun-Nov) with daily persistence.
    """
    np.random.seed(seed)
    
    start_date = pd.to_datetime(start)
    end_date = start_date + pd.DateOffset(years=years) - pd.Timedelta(hours=1)
    df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq="h"))
    
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    df["doy"] = df.index.dayofyear
    
    # 1. Recalibrated Seasonal Climate Factor (Centered perfectly for Manaus)
    # Floor (0.0) lands around Feb/Mar (Heavy rains); Ceiling (1.0) hits around Aug/Sep (Peak Dry)
    df["season_factor"] = 0.5 * (1 - np.cos(2 * np.pi * (df["doy"] - 110) / 365))
    
    # Weather state transition matrices (Applied at daily scale for systemic persistence)
    # Wet season: high probability of staying rainy or partial
    t_wet = np.array([[0.25, 0.45, 0.30], [0.10, 0.50, 0.40], [0.05, 0.25, 0.70]])
    # Dry season: high probability of staying sunny
    t_dry = np.array([[0.85, 0.12, 0.03], [0.45, 0.45, 0.10], [0.30, 0.40, 0.30]])
    states = ["sunny", "partial", "rainy"]
    
    # Generate daily base states to ensure "days of consecutive rain/sun"
    num_days = len(df.resample("D"))
    daily_doy = df.resample("D").first()["doy"].values
    daily_season_factor = 0.5 * (1 - np.cos(2 * np.pi * (daily_doy - 110) / 365))
    
    daily_weather = ["partial"]
    for i in range(1, num_days):
        current_idx = states.index(daily_weather[-1])
        sf = daily_season_factor[i]
        p_interp = (1 - sf) * t_wet[current_idx] + sf * t_dry[current_idx]
        p_interp /= p_interp.sum()
        
        daily_weather.append(np.random.choice(states, p=p_interp))
    
    # Broadcast daily states back to hourly rows
    df["weather_day"] = np.repeat(daily_weather, 24)[:len(df)]
    
    # Hourly weather factors (Rainy days in winter severely suppress irradiance baseline)
    factors = {"sunny": 1.0, "partial": 0.50, "rainy": 0.20}
    df["weather_factor"] = df["weather_day"].map(factors)
    
    # 2. Dynamic Solar Clearing Ceiling with Daily Variability
    df["g_max_base"] = 680 + (g_max_dry - 680) * df["season_factor"]
    
    fator_diario = np.random.normal(1.0, 0.05, num_days)
    df["fator_diario"] = np.repeat(fator_diario, 24)[:len(df)]
    df["g_max_dinamico"] = df["g_max_base"] * df["fator_diario"]
    
    # Astronomical Geometry (Manaus, Lat -3.11)
    declination = 23.45 * np.sin(2 * np.pi * (284 + df["doy"]) / 365) * (np.pi / 180)
    latitude = -3.11 * (np.pi / 180)
    hour_angle = (df["hour"] - 12) * 15 * (np.pi / 180)
    
    cos_zenith = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    solar_profile = np.clip(cos_zenith, 0, None)
    
    g_clear = solar_profile * df["g_max_dinamico"]
    g_with_weather = g_clear * df["weather_factor"]
    
    # Noise profiles scaled down to avoid unphysical spikes
    mult_wet = df["weather_day"].map({"sunny": 0.2, "partial": 0.6, "rainy": 0.1})
    mult_dry = df["weather_day"].map({"sunny": 0.05, "partial": 0.3, "rainy": 0.15})
    df["noise_mult"] = (1 - df["season_factor"]) * mult_wet + df["season_factor"] * mult_dry
    
    base_noise = np.random.normal(0, 1, len(df)) * (df["noise_mult"] * noise_std)
    base_noise = base_noise * (solar_profile > 0)
    
    # Final Output Irradiance Block
    is_astronomical_night = solar_profile == 0
    g_noisy = g_with_weather + base_noise
    
    g_threshold = 45.0
    is_dark_or_twilight = (g_with_weather < g_threshold) | is_astronomical_night
    
    df["G"] = np.where(is_astronomical_night, 0.0, np.clip(g_noisy, 0.0, 1350.0))
    df.loc[is_dark_or_twilight, "G"] = 0.0
    
    # 3. Correct Nocturnal Terminology Mapping
    df["weather"] = df["weather_day"]
    df.loc[is_dark_or_twilight & (df["weather_day"] == "sunny"), "weather"] = "clear_night"
    df.loc[is_dark_or_twilight & (df["weather_day"] == "partial"), "weather"] = "cloudy_night"
    df.loc[is_dark_or_twilight & (df["weather_day"] == "rainy"), "weather"] = "rainy_night"
    

    # Clean up working columns
    df = df.drop(columns=["weather_day"])
    
    return df


def simulate_military_demand(df, peak_demand_w=50000, daily_vol=0.25, hourly_vol=0.15, seed=42):
    """
    Simulates operational energy demand for an isolated military facility (e.g., PEF).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing 'season_factor', 'hour', and 'G'.
    peak_demand_w : float
        The theoretical maximum peak power demand of the facility.
    daily_vol : float
        Standard deviation for daily operational scaling (inter-day randomness).
    hourly_vol : float
        Standard deviation for hourly operational noise (intra-day volatility).
    seed : int
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    n = len(df)
    
    df["weekday"] = df.index.weekday
    df["is_weekend"] = np.where(df["weekday"].isin([5, 6]), True, False)
    
    # 1. ELEVATED BASELINE LOAD (24/7 Radar, Comms, and Surveillance Systems)
    seasonal_base = (0.35 + (0.42 - 0.35) * df["season_factor"]) * peak_demand_w
    
    # Tactical Operational Profiles (Gaussian Mixture Model Bases)
    hours = np.arange(24)
    profile_workday = 0.65 * np.exp(-((hours - 10) / 2.5) ** 2) + 0.55 * np.exp(-((hours - 15) / 2.0) ** 2) + 0.05
    profile_weekend = 0.35 * np.exp(-((hours - 11) / 3.0) ** 2) + 0.40 * np.exp(-((hours - 16) / 2.5) ** 2) + 0.05
    
    ops_workday = df["hour"].map(pd.Series(profile_workday, index=hours))
    ops_weekend = df["hour"].map(pd.Series(profile_weekend, index=hours))
    
    # --- CONTROLLABLE STOCHASTIC OPERATIONAL VARIABILITY ---
    # Inter-day variance (Daily structural shift)
    num_days = len(df.resample("D"))
    daily_ops_modifier = np.random.normal(1.0, daily_vol, num_days)
    df["daily_ops_modifier"] = np.repeat(daily_ops_modifier, 24)[:n]
    
    # Intra-day variance (Hourly high-frequency chattering)
    hourly_ops_noise = np.random.normal(1.0, hourly_vol, n)
    
    # Modulated stochastic active operational load
    base_ops_profile = np.where(df["is_weekend"], ops_weekend, ops_workday)
    df["operational_load"] = (base_ops_profile * (peak_demand_w * 0.4) * df["daily_ops_modifier"] * hourly_ops_noise)
    df["operational_load"] = np.clip(df["operational_load"], 0, None) 

    # 2. THERMAL LOAD MODELING (Critical Data Rooms, Server Racks, Ammo Depot)
    thermal_load = np.zeros(n)
    alpha_gain = 0.00042  
    beta_loss = 0.22      
    
    g_values = df["G"].values
    s_factors = df["season_factor"].values
    
    for t in range(1, n):
        fator_termico_dinamico = 0.90 + (1.10 - 0.90) * s_factors[t]
        carga_retida = thermal_load[t-1] * (1 - beta_loss)
        ganho_solar = alpha_gain * g_values[t]
        thermal_load[t] = (carga_retida + ganho_solar) * fator_termico_dinamico
    
    if thermal_load.max() > 0:
        thermal_load = (thermal_load / thermal_load.max()) * (peak_demand_w * 0.30)
    
    df["thermal_load"] = thermal_load

    # 3. STOCHASTIC STRESS AND NOISE (Grid surges, secondary subsystems)
    random_noise = np.random.normal(0, peak_demand_w * 0.05, n)
    
    # Total tactical load compilation
    demand = seasonal_base + df["operational_load"] + df["thermal_load"] + random_noise
    
    # Strict physical defense floor (30% of peak_demand_w)
    df["Demand"] = np.clip(demand, peak_demand_w * 0.3, None)
    
    # Clean up temporary structural columns
    df = df.drop(columns=["daily_ops_modifier"])
    
    return df


def calculate_pure_solar_power(irradiance, num_panels, panel_efficiency=0.21, panel_area=2.0):
    total_area = num_panels * panel_area    
    power_produced = irradiance * total_area * panel_efficiency
    return power_produced


class BESS:
    def __init__(self, cap_energy_wh, pot_max_w, initial_soc_pct=1.0):
        self.cap_energy_wh = cap_energy_wh
        self.pot_max_w = pot_max_w
        self.charge_efficiency = 0.92
        self.discharge_efficiency = 0.92
        self.min_energy_wh = self.cap_energy_wh * 0.10
        self.current_energy_wh = self.cap_energy_wh * initial_soc_pct


    def get_soc_pct(self):
        return (self.current_energy_wh / self.cap_energy_wh) * 100.0


    def update_hourly(self, power_request_w):
        power_request_w = max(-self.pot_max_w, min(power_request_w, self.pot_max_w))
        actual_power_w = 0.0

        if power_request_w > 0: # LOAD
            energy_to_add = power_request_w * self.charge_efficiency
            if self.current_energy_wh + energy_to_add <= self.cap_energy_wh:
                self.current_energy_wh += energy_to_add
                actual_power_w = power_request_w
            else:
                space = self.cap_energy_wh - self.current_energy_wh
                self.current_energy_wh = self.cap_energy_wh
                actual_power_w = space / self.charge_efficiency
        
        elif power_request_w < 0: # DELOAD
            energy_needed = abs(power_request_w) / self.discharge_efficiency
            if self.current_energy_wh - energy_needed >= self.min_energy_wh:
                self.current_energy_wh -= energy_needed
                actual_power_w = power_request_w
            else:
                available = self.current_energy_wh - self.min_energy_wh
                self.current_energy_wh = self.min_energy_wh
                actual_power_w = -(available * self.discharge_efficiency)
        
        return actual_power_w, (power_request_w - actual_power_w)


    def get_status(self):
        return {
            "soc_pct": self.get_soc_pct(),
            "energy_available_wh": self.current_energy_wh - self.min_energy_wh,
            "is_available": self.current_energy_wh > self.min_energy_wh,
            "pot_max_w": self.pot_max_w
        }


class HydrogenStorage:
    def __init__(self, cap_energy_wh, pot_max_w, initial_soc_pct=1.0):
        self.cap_energy_wh = cap_energy_wh
        self.pot_max_w = pot_max_w
        self.charge_efficiency = 0.65
        self.discharge_efficiency = 0.55
        self.min_energy_wh = self.cap_energy_wh * 0.05
        self.current_energy_wh = self.cap_energy_wh * initial_soc_pct

    def get_soc_pct(self):
        return (self.current_energy_wh / self.cap_energy_wh) * 100.0
    
    def update_hourly(self, power_request_w):
        power_request_w = max(-self.pot_max_w, min(power_request_w, self.pot_max_w))
        actual_power_w = 0.0

        if power_request_w > 0: # CARGA
            energy_to_add = power_request_w * self.charge_efficiency
            if self.current_energy_wh + energy_to_add <= self.cap_energy_wh:
                self.current_energy_wh += energy_to_add
                actual_power_w = power_request_w
            else:
                space = self.cap_energy_wh - self.current_energy_wh
                self.current_energy_wh = self.cap_energy_wh
                actual_power_w = space / self.charge_efficiency
        
        elif power_request_w < 0: # DESCARGA
            energy_needed = abs(power_request_w) / self.discharge_efficiency
            if self.current_energy_wh - energy_needed >= self.min_energy_wh:
                self.current_energy_wh -= energy_needed
                actual_power_w = power_request_w
            else:
                available = self.current_energy_wh - self.min_energy_wh
                self.current_energy_wh = self.min_energy_wh
                actual_power_w = -(available * self.discharge_efficiency)
        
        return actual_power_w, (power_request_w - actual_power_w)

    def get_status(self):
        return {
            "soc_pct": self.get_soc_pct(),
            "energy_available_wh": self.current_energy_wh - self.min_energy_wh,
            "is_available": self.current_energy_wh > self.min_energy_wh,
            "pot_max_w": self.pot_max_w
        }


class DieselGenerator:
    def __init__(self, capacity_w, fuel_tank_liters, consumption_rate_l_per_kwh=0.25):
        self.capacity_w = capacity_w  
        self.fuel_tank_liters = fuel_tank_liters  
        self.current_fuel_liters = fuel_tank_liters
        self.consumption_rate_l_per_wh = consumption_rate_l_per_kwh / 1000.0

    def update_hourly(self, needed_power_w):
        power_to_generate_w = min(needed_power_w, self.capacity_w)
        fuel_used = power_to_generate_w * self.consumption_rate_l_per_wh
        
        if self.current_fuel_liters >= fuel_used:
            self.current_fuel_liters -= fuel_used
            return power_to_generate_w, fuel_used
        else:
            actual_power_w = (self.current_fuel_liters / self.consumption_rate_l_per_wh)
            fuel_used = self.current_fuel_liters
            self.current_fuel_liters = 0
            return actual_power_w, fuel_used

    def get_status(self):
        return {
            "fuel_level_l": self.current_fuel_liters,
            "is_available": self.current_fuel_liters > 0,
            "capacity_w": self.capacity_w
        }


def plot_microgrid_3columns(df):
    """
    Generates a structured 3x3 subplot matrix for the hybrid microgrid (PEF).
    All labels, titles, and units are formatted in English for academic publication.
    """
    # Ensure output directory exists using the 'os' module
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create a 3x3 matrix of subplots (sharex=True ensures time synchronization)
    fig, axs = plt.subplots(3, 3, figsize=(22, 12), sharex=True)
        
    # ==========================================================================
    # COLUMN 1: DEMAND AND PRIMARY POWER BALANCE (DYNAMIC FLOWS)
    # ==========================================================================
    
    # 1. Required Load Demand
    axs[0, 0].plot(df.index, df['Demand'], label='Demand (W)', color='crimson', lw=1.2)
    axs[0, 0].set_ylabel('Power (W)')
    axs[0, 0].set_title('1. Required Power Demand', fontsize=11, fontweight='bold')
    axs[0, 0].grid(True, linestyle='--', alpha=0.4)
    axs[0, 0].legend(loc='upper right')
        
    # 2. Net Photovoltaic Generation
    axs[1, 0].plot(df.index, df['SOLAR_POWER'], label='PV Generation', color='forestgreen', lw=1.2)
    axs[1, 0].set_ylabel('Power (W)')
    axs[1, 0].set_title('2. Net Photovoltaic Generation', fontsize=11, fontweight='bold')
    axs[1, 0].grid(True, linestyle='--', alpha=0.4)
    axs[1, 0].legend(loc='upper right')
    
    # 3. Dynamic Net Power Balance
    axs[2, 0].plot(df.index, df['NET_POWER'], label='Net Power', color='dodgerblue', lw=1)
    axs[2, 0].axhline(0, color='black', linestyle='-', alpha=0.3)
    axs[2, 0].set_ylabel('Power (W)')
    axs[2, 0].set_xlabel('Time Horizon (Hours)')
    axs[2, 0].set_title('3. Instantaneous Net Balance & Deficit Risk', fontsize=11, fontweight='bold')
    axs[2, 0].grid(True, linestyle='--', alpha=0.4)
    axs[2, 0].legend(loc='upper right')

    # ==========================================================================
    # COLUMN 2: ACTIVE ASSET DISPATCH & BACKUP (DYNAMIC FLOWS)
    # ==========================================================================
    
    # 4. BESS Power Dispatch (Charge/Discharge)
    axs[0, 1].plot(df.index, df['BESS_POWER_W'], label='BESS Power', color='purple', lw=1.2)
    axs[0, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axs[0, 1].set_ylabel('Power (W)')
    axs[0, 1].set_title('4. Dynamic BESS Dispatch', fontsize=11, fontweight='bold')
    axs[0, 1].grid(True, linestyle='--', alpha=0.4)
    axs[0, 1].legend(loc='upper right')
    
    # 5. Hydrogen System Power Dispatch (Electrolysis/Fuel Cell)
    axs[1, 1].plot(df.index, df['H2_POWER_W'], label='H2 Power', color='teal', lw=1.2)
    axs[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axs[1, 1].set_ylabel('Power (W)')
    axs[1, 1].set_title('5. Hydrogen System Dispatch', fontsize=11, fontweight='bold')
    axs[1, 1].grid(True, linestyle='--', alpha=0.4)
    axs[1, 1].legend(loc='upper right')
    
    # 6. Backup Diesel Generator Dispatch
    axs[2, 1].plot(df.index, df['DIESEL_USAGE_W'], label='Diesel Power', color='darkorange', lw=1.2)
    axs[2, 1].set_ylabel('Power (W)')
    axs[2, 1].set_xlabel('Time Horizon (Hours)')
    axs[2, 1].set_title('6. Diesel Generator Operation', fontsize=11, fontweight='bold')
    axs[2, 1].grid(True, linestyle='--', alpha=0.4)
    axs[2, 1].legend(loc='upper right')

    # ==========================================================================
    # COLUMN 3: ENERGY STATES (SoC), FUEL INVENTORY & RISK MANAGEMENT
    # ==========================================================================
    # 7. BESS State of Charge (Stored Energy)
    axs[0, 2].plot(df.index, df['BESS_SOC_WH'], label='BESS Energy', color='orchid', lw=1.5)
    axs[0, 2].set_ylabel('Capacity (Wh)')
    axs[0, 2].set_title('7. BESS Energy Level (SoC)', fontsize=11, fontweight='bold')
    axs[0, 2].grid(True, linestyle='--', alpha=0.4)
    axs[0, 2].legend(loc='upper right')
    
    # 8. Hydrogen Storage Tank Level
    axs[1, 2].plot(df.index, df['H2_SOC_WH'], label='H2 Energy', color='darkcyan', lw=1.5)
    axs[1, 2].set_ylabel('Capacity (Wh)')
    axs[1, 2].set_title('8. Seasonal H₂ Tank Inventory', fontsize=11, fontweight='bold')
    axs[1, 2].grid(True, linestyle='--', alpha=0.4)
    axs[1, 2].legend(loc='upper right')
    
    # 9. Fuel Inventory and Risk Analysis (Unmet Load)
    axs[2, 2].plot(df.index, df['DIESEL_TANK_L'], label='Diesel Stock', color='darkred', lw=1.5)
    axs[2, 2].set_ylabel('Fuel (Liters)')
    axs[2, 2].set_xlabel('Time Horizon (Hours)')
    axs[2, 2].set_title('9. Logistic Fuel Stock Inventory', fontsize=11, fontweight='bold')
    axs[2, 2].grid(True, linestyle='--', alpha=0.4)
    axs[2, 2].legend(loc='upper left')

    # Global alignment and scientific high-resolution saving
    plt.suptitle('Time-Series Operational Dispatch Analysis of the Hybrid Microgrid - Amazon PEF', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join("outputs", "microgrid_3columns_subplot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"3-column subplot successfully updated and saved to: {output_path}")


def plot_environmental_and_operational_analysis(df):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True)

    df["G"].plot(ax=ax1,style=".",markersize=2,alpha=0.6,color="#2ca02c")
    ax1.set_title("COUPLED ATMOSPHERIC & OPERATIONAL ANALYSIS - ISOLATED MILITARY OUTPOST (AMAZONIA)",fontsize=14,fontweight="bold",pad=15)
    ax1.set_ylabel("Solar Irradiance G\n[W/m²]",fontsize=11,fontweight="bold")
    ax1.grid(True,alpha=0.2,linestyle="--")
    ax1.legend(["Hourly Irradiance (G)"],loc="upper right")

    df["Demand"].plot(ax=ax2,style=".",markersize=2,alpha=0.6,color="#1f77b4")
    ax2.set_ylabel("Tactical Power Demand\n[kW]",fontsize=11,fontweight="bold")
    ax2.set_xlabel("Time Series Timeline (Date)",fontsize=12,fontweight="bold")
    ax2.grid(True,alpha=0.2,linestyle="--")
    ax2.legend(["Total System Demand"],loc="upper right")

    plt.tight_layout()
    plt.show()