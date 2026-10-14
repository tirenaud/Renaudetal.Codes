#%%
import pandas as pd
from utils import load_input


def getIVESPAtable():
    # returns a usable version of the IVESPA database, the same one as in the geneve model code
    # exact copy of the function used in geneve model

    options = {
        'dtype': {'30': float, '32': float, '55': float, '56': float, '57': float, '58': float},
        'engine': 'openpyxl' 
    }

    init_dir = load_input() #Drive data folder

    # link to the IVESPA csv file
    ivespa = pd.read_excel(f"{init_dir}/ATMOSPHERIC PROFILES/IVESPAvn1.0.xlsx", **options, header=0)

    ivespa_ID = ivespa['IVESPA ID'].astype(str)

    vh0 = ivespa['Vent altitude (m a.s.l.)'].values

    dur = ivespa['Duration Best estimate (hours)'].values

    tem = ivespa['TEM Best estimate (kg)'].values

    top_height = ivespa['Tephra Plume Top (km asl) Best estimate'].values

    spreading_height = ivespa['Spreading height of the Umbrella Cloud (km asl) Best estimate'].values

    top_height_un = ivespa['Tephra Plume Top (km asl) Uncertainty'].values

    spreading_height_un = ivespa['Spreading height of  the Umbrella Cloud (km asl) Uncertainty'].values

    d0 = ivespa['Vent Diameter (m)\t Best estimate'].values

    d0_un = ivespa['Vent Diameter (m)\t Uncertainty'].values

    u0 = ivespa['Exit velocity (m/s) Best estimate'].values

    u0_un = ivespa['Exit velocity (m/s) Uncertainty'].values

    dur_un = ivespa['Duration Uncertainty (hours)'].values

    tem_un_high = ivespa['TEM Uncertainty (high; kg)'].values

    tem_un_low = ivespa['TEM Uncertainty (low; kg)'].values

    average_wind_NOAA = ivespa['Average wind speed (m/s, NOAA reanalyses)'].values

    average_wind_ERA = ivespa['Average wind speed (m/s, ERA reanalyses)'].values

    BVfrequency_NOAA = ivespa['Average Brunt-Väisälä frequency (/s, NOAA reanalyses)'].values

    BVfrequency_ERA = ivespa['Average Brunt-Väisälä frequency (/s, ERA reanalyses)'].values

    TGSD_available = ivespa['TGSD Available'].astype(str)

    latitude = ivespa['Latitude (deg. N)'].values

    style = ivespa['Eruption style'].astype(str)

    morphology = ivespa['Plume morphology'].astype(str)

    method = ivespa['Tephra Plume Top Method'].astype(str)

    year = ivespa['Event Year'].values

    ivespa_tbl = pd.DataFrame({
        'Ivespa_ID': ivespa_ID,
        'VentAltitude': vh0,
        'Duration': dur,
        'Duration_Uncertainty': dur_un,
        'TEM': tem,
        'TEM_UncertaintyHigh': tem_un_high,
        'TEM_UncertaintyLow': tem_un_low,
        'TopHeight': top_height,
        'TopHeight_Uncertainty': top_height_un,
        'SpreadingHeight': spreading_height,
        'SpreadingHeight_Uncertainty': spreading_height_un,
        'VentDiameter': d0,
        'VentDiameter_Uncertainty': d0_un,
        'ExitVelocity': u0,
        'ExitVelocity_Uncertainty': u0_un,
        'Average_Wind_Speed_NOAA': average_wind_NOAA,
        'Average_Wind_Speed_ERA': average_wind_ERA,
        'Average_Brunt_Vaisala_Frequency_NOAA': BVfrequency_NOAA,
        'Average_Brunt_Vaisala_Frequency_ERA': BVfrequency_ERA,
        'TGSDAvailable': TGSD_available,
        'LatitudeN': latitude,
        'Eruption_Style': style,
        'Plume_Morphology': morphology,
        'Method': method,
        'Year': year
    })


    ivespa_tbl_units = {
        'VentAltitude': '-',
        'Duration': 'hours',
        'Duration_Uncertainty': 'hours',
        'TEM': 'kg',
        'TEM_UncertaintyHigh': 'kg',
        'TEM_UncertaintyLow': 'kg',
        'TopHeight': 'km a.s.l.',
        'TopHeight_Uncertainty': 'km a.s.l.',
        'SpreadingHeight': 'km a.s.l.',
        'SpreadingHeight_Uncertainty': 'km a.s.l.',
        'VentDiameter': 'm',
        'VentDiameter_Uncertainty': 'm',
        'ExitVelocity': 'm/s',
        'ExitVelocity_Uncertainty': 'm/s',
        'Average_Wind_Speed_NOAA': 'm/s',
        'Average_Wind_Speed_ERA': 'm/s',
        'Average_Brunt_Vaisala_Frequency_NOAA': '/s',
        'Average_Brunt_Vaisala_Frequency_ERA': '/s',
        'TGSDAvailable': '-',
        'LatitudeN': 'degN',
        'Eruption_Style': '-',
        'Plume_Morphology': '-',
        'Method': '-',
        'Year': '-'
    }

    # assigning the units
    for col, unit in ivespa_tbl_units.items():
        ivespa_tbl[col] = pd.Series(ivespa_tbl[col], dtype='object')
        ivespa_tbl[col].units = unit

    return ivespa_tbl

