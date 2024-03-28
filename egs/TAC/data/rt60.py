"""
#######################
#pip install acoustics#
#######################
Example:

#Produces 6 octave bands array([ 125,  250,  500, 1000, 2000, 4000])
req_bands = np.array([125 * pow(2,a) for a in range(6)]

t60_impulse(RIR, 16000, req_bands)

"""
import numpy as np
from scipy import signal
from scipy.fft import fftfreq,fft
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.signal import bandpass
from scipy import stats





def t60_impulse(raw_signal, fs, bands, rt='t10'):  # pylint: disable=too-many-locals
    """
    Returns RT_60 in octave bands on different estimators t30, t20, t10.
    
    Reverberation time from a WAV impulse response.
    :param raw_signal: Room impulse response as NumPy array.
    :param bands: Octave or third bands as NumPy array. 6 Octave bands example  -> req_bands = np.array([125 * pow(2,a) for a in range(6)]) 
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`
    """
    #fs, raw_signal = wavfile.read(file_name)
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]

        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    return t60
