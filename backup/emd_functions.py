import numpy as np
from scipy import interpolate
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq

def find_extrema(signal):
    maxima = []
    minima = []
    
    N = len(signal)
    for i in range(1, N - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            maxima.append(i)
        elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            minima.append(i)
    
    return np.array(maxima), np.array(minima)


def get_envelopes(signal, maxima, minima):
    t = np.arange(len(signal))
    
    if len(maxima) < 2 or len(minima) < 2:
        return None, None
    
    max_prepend = 2 * maxima[0] - maxima[1] if maxima[0] > 0 else 0
    max_append = 2 * maxima[-1] - maxima[-2] if maxima[-1] < len(signal)-1 else len(signal)-1
    
    min_prepend = 2 * minima[0] - minima[1] if minima[0] > 0 else 0
    min_append = 2 * minima[-1] - minima[-2] if minima[-1] < len(signal)-1 else len(signal)-1
    
    ext_max_idx = np.concatenate([[max_prepend], maxima, [max_append]])
    ext_max_val = np.concatenate([[signal[maxima[0]]], signal[maxima], [signal[maxima[-1]]]])
    
    ext_min_idx = np.concatenate([[min_prepend], minima, [min_append]])
    ext_min_val = np.concatenate([[signal[minima[0]]], signal[minima], [signal[minima[-1]]]])
    
    try:
        upper_envelope_func = interpolate.CubicSpline(ext_max_idx, ext_max_val, bc_type='natural')
        lower_envelope_func = interpolate.CubicSpline(ext_min_idx, ext_min_val, bc_type='natural')
        
        upper_envelope = upper_envelope_func(t)
        lower_envelope = lower_envelope_func(t)
        
        return upper_envelope, lower_envelope
    
    except:
        return None, None


def sift(signal, max_iter=10, sd_threshold=0.3):
    h = signal.copy()
    
    for iteration in range(max_iter):
        h_prev = h.copy()
        
        # Find extrema
        maxima, minima = find_extrema(h)
        
        # Check stopping criteria - not enough extrema
        if len(maxima) < 2 or len(minima) < 2:
            break
        
        # Get envelopes
        upper_env, lower_env = get_envelopes(h, maxima, minima)
        
        if upper_env is None or lower_env is None:
            break
        
        # Calculate mean envelope
        mean_env = (upper_env + lower_env) / 2
        
        # Subtract mean
        h = h - mean_env
        
        # Calculate stopping criterion (SD)
        if iteration > 0:
            sd = np.sum(((h_prev - h) ** 2) / (h_prev ** 2 + 1e-10))
            
            if sd < sd_threshold:
                break
    
    return h


def is_imf(signal):
    maxima, minima = find_extrema(signal)
    
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    num_extrema = len(maxima) + len(minima)
    
    if abs(num_extrema - zero_crossings) > 1:
        return False
    
    # Condition 2: mean of envelopes should be close to zero
    if len(maxima) >= 2 and len(minima) >= 2:
        upper_env, lower_env = get_envelopes(signal, maxima, minima)
        if upper_env is not None and lower_env is not None:
            mean_env = (upper_env + lower_env) / 2
            if np.mean(np.abs(mean_env)) > 0.1 * np.std(signal):
                return False
    
    return True


def emd(signal, max_imfs=10, max_iter=10, sd_threshold=0.3):
   
    imfs = []
    residual = signal.copy()
    
    for i in range(max_imfs):
        # Check if residual is monotonic (stopping criterion)
        maxima, minima = find_extrema(residual)
        
        if len(maxima) < 2 or len(minima) < 2:
            break
        
        # Extract IMF
        imf = sift(residual, max_iter=max_iter, sd_threshold=sd_threshold)
        
        imfs.append(imf)
        residual = residual - imf
        
        # Stop if residual becomes too small
        if np.std(residual) < 0.001 * np.std(signal):
            break
    
    return imfs, residual


def compute_instantaneous_frequency(imf, fs):
    # Hilbert transform
    analytic_signal = hilbert(imf)
    
    # Instantaneous amplitude
    inst_amp = np.abs(analytic_signal)
    
    # Instantaneous phase
    inst_phase = np.unwrap(np.angle(analytic_signal))
    
    # Instantaneous frequency (derivative of phase)
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])  
    inst_freq[inst_freq < 0] = 0
    
    return inst_freq, inst_amp


def compute_dominant_frequency(imf, fs):

    N = len(imf)
    
    yf = fft(imf)
    xf = fftfreq(N, 1/fs)
    
    positive_freq_idx = xf > 0
    xf_positive = xf[positive_freq_idx]
    yf_positive = np.abs(yf[positive_freq_idx])
    
    dominant_idx = np.argmax(yf_positive)
    dominant_freq = xf_positive[dominant_idx]
    
    return dominant_freq, xf_positive, yf_positive


def analyze_imf(imf, fs):
    # Find extrema
    maxima, minima = find_extrema(imf)
    
    # Get envelopes
    upper_env, lower_env = get_envelopes(imf, maxima, minima)
    mean_env = None
    if upper_env is not None and lower_env is not None:
        mean_env = (upper_env + lower_env) / 2
    
    # Instantaneous frequency
    inst_freq, inst_amp = compute_instantaneous_frequency(imf, fs)
    
    # Dominant frequency
    dominant_freq, freq_spectrum, magnitude_spectrum = compute_dominant_frequency(imf, fs)
    
    # Statistics
    energy = np.sum(imf ** 2)
    std_dev = np.std(imf)
    mean_val = np.mean(imf)
    max_amp = np.max(np.abs(imf))
    
    # Zero crossings
    zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
    
    analysis = {
        'maxima': maxima,
        'minima': minima,
        'upper_envelope': upper_env,
        'lower_envelope': lower_env,
        'mean_envelope': mean_env,
        'inst_freq': inst_freq,
        'inst_amp': inst_amp,
        'dominant_freq': dominant_freq,
        'freq_spectrum': freq_spectrum,
        'magnitude_spectrum': magnitude_spectrum,
        'energy': energy,
        'std_dev': std_dev,
        'mean': mean_val,
        'max_amplitude': max_amp,
        'zero_crossings': zero_crossings,
        'num_maxima': len(maxima),
        'num_minima': len(minima)
    }
    
    return analysis


def calculate_respiratory_rate_from_imf(imf, fs, expected_range=(0.1, 0.5)):
    dominant_freq, freq_spectrum, magnitude_spectrum = compute_dominant_frequency(imf, fs)
    
    if expected_range[0] <= dominant_freq <= expected_range[1]:
        respiratory_rate = dominant_freq * 60
    else:
        inst_freq, inst_amp = compute_instantaneous_frequency(imf, fs)
        mask = (inst_freq >= expected_range[0]) & (inst_freq <= expected_range[1])
        
        if np.sum(mask) > 0:
            valid_freqs = inst_freq[mask]
            valid_amps = inst_amp[mask]
            mean_freq = np.average(valid_freqs, weights=valid_amps)
            respiratory_rate = mean_freq * 60
        else:
            respiratory_rate = dominant_freq * 60
    
    return respiratory_rate, dominant_freq

def select_respiratory_imf(imfs, fs, expected_range=(0.1, 0.5)):
    best_score = 0
    best_idx = 0
    best_rr = 0
    
    for idx, imf in enumerate(imfs):
        dominant_freq, _, magnitude_spectrum = compute_dominant_frequency(imf, fs)
        
        inst_freq, inst_amp = compute_instantaneous_frequency(imf, fs)
        mask = (inst_freq >= expected_range[0]) & (inst_freq <= expected_range[1])
        energy_in_range = np.sum(inst_amp[mask] ** 2)
        total_energy = np.sum(inst_amp ** 2)
        
        energy_ratio = energy_in_range / (total_energy + 1e-10)
        
        freq_in_range = 1.0 if (expected_range[0] <= dominant_freq <= expected_range[1]) else 0.3
        
        score = energy_ratio * freq_in_range
        
        if score > best_score:
            best_score = score
            best_idx = idx
            best_rr = dominant_freq * 60
    
    return best_idx, best_rr