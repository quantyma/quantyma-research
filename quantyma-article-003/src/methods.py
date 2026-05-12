import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

from pathlib import Path

from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_all_audio_files_for_file_set(set_path, sr) -> dict:

    all_files = [f for f in set_path.iterdir() if f.suffix == '.wav']

    data = {}
    for f in all_files:
        y, sr = sf.read(f)
        data[f.stem] = y
    
    return data 


def full_noise_profiles_plot(audio_dict, sr=16000, save_path=None, dpi=300):
    """
    Analyzes and plots the Power Spectral Density (PSD) for different noise categories.
    Includes individual samples, the category mean, and the standard deviation shadow.
    """
    plt.figure(figsize=(12, 6))
    
    # Define fixed colors for each noise level
    colors = {'low': 'blue', 'medium': 'orange', 'high': 'green'}

    for category in ['low', 'medium', 'high']:
        # Filter keys belonging to the current category
        keys = [k for k in audio_dict.keys() if k.startswith(category)]
        
        if not keys:
            continue
            
        category_psds = []
        
        # Process each audio sample in the category
        for k in keys:
            y = audio_dict[k]
            # Calculate PSD using Welch's method (fs=16000Hz)
            freqs, psd = signal.welch(y, fs=sr, nperseg=1024)
            # Convert power to decibels (dB/Hz)
            psd_db = 10 * np.log10(psd + 1e-12)
            category_psds.append(psd_db)
            
            # Plot individual background lines (thin and transparent)
            plt.plot(freqs, psd_db, color=colors[category], alpha=0.1, linewidth=0.5)

        # Statistical calculations
        category_psds = np.array(category_psds)
        mean_psd = np.mean(category_psds, axis=0)
        std_psd = np.std(category_psds, axis=0)

        # Plot Standard Deviation shadow (Confidence Interval)
        plt.fill_between(freqs, 
                         mean_psd - std_psd, 
                         mean_psd + std_psd, 
                         color=colors[category], 
                         alpha=0.2)

        # Plot the thick Mean line for the category
        plt.plot(freqs, mean_psd, color=colors[category], linewidth=2, label=f'Mean: {category}')

    # Final plot styling
    plt.title("Statistical Frequency Analysis: Mean PSD vs Variability")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    
    # Legend handling
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')


    plt.show()


def plot_average_spectrograms(audio_dict, sr=16000, save_path=None, dpi=300):
    """
    Computes and plots the AVERAGE spectrogram for all samples in each category.
    This provides a robust statistical signature of the noise levels.
    """
    categories = ['low', 'medium', 'high']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, category in enumerate(categories):
        keys = [k for k in audio_dict.keys() if k.startswith(category)]
        
        if not keys:
            continue

        all_sxx = []
        
        # Collect spectrograms for all files in this category
        for k in keys:
            y = audio_dict[k]
            f, t, Sxx = signal.spectrogram(y, fs=sr)
            all_sxx.append(Sxx)
        
        # Calculate the Mean Spectrogram (Energy domain)
        avg_sxx = np.median(all_sxx, axis=0)
        
        # Convert the average to dB for visualization
        avg_sxx_db = 10 * np.log10(avg_sxx + 1e-12)
        
        im = axes[i].pcolormesh(t, f, avg_sxx_db, shading='gouraud', cmap='magma')
        axes[i].set_title(f"Average Spectrogram: {category} (N={len(keys)})")
        axes[i].set_xlabel("Time (s)")
        
        if i == 0: 
            axes[i].set_ylabel("Frequency (Hz)")
            
        plt.colorbar(im, ax=axes[i], label='Average dB')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def get_detailed_eda_df(audio_dict, save_path=None):
    stats_list = []

    for key, signal_data in audio_dict.items():
        rms = np.sqrt(np.mean(np.square(signal_data)))
        peak = np.max(np.abs(signal_data))
        crest = peak / (rms + 1e-12)
        kurt = kurtosis(signal_data, fisher=True)

        category = key.split('_')[0].lower()

        stats_list.append({
            'Noise Category': category,
            'RMS': rms,
            'Crest Factor': crest,
            'Kurtosis': kurt,
        })

    df_results = pd.DataFrame(stats_list)

    df_summary = df_results.groupby('Noise Category').agg(['mean', 'std'])

    order = ['low', 'medium', 'high']
    df_summary = df_summary.reindex(order)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df_summary.columns = [
                                'RMS_mean', 'RMS_std',
                                'CF_mean', 'CF_std',
                                'Kurtosis_mean', 'Kurtosis_std'
                            ]
        df_summary.to_csv(save_path)


    return df_summary


def extract_features(y, sr):

    freqs, psd = signal.welch(y, fs=sr, nperseg=1024)

    analytic = signal.hilbert(y)
    envelope = np.abs(analytic)

    peaks, _ = signal.find_peaks(y, distance=sr*0.001)
    peak_vals = y[peaks] if len(peaks) > 0 else np.array([0])

    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / sr
    else:
        peak_intervals = np.array([0])

    return {
        "rms": np.sqrt(np.mean(y**2)),
        "energy": np.sum(y**2),
        "kurtosis": kurtosis(y),
        "skewness": skew(y),
        "zero_crossing": np.mean(np.diff(np.sign(y)) != 0),

        "psd_total": np.sum(psd),
        "spectral_centroid": np.sum(freqs * psd) / (np.sum(psd) + 1e-12),
        "dominant_freq": freqs[np.argmax(psd)],

        "env_mean": np.mean(envelope),
        "env_std": np.std(envelope),

        "num_peaks": len(peaks),
        "peak_mean": np.mean(peak_vals),
        "peak_std": np.std(peak_vals),

        "peak_interval_mean": np.mean(peak_intervals),
        "peak_interval_std": np.std(peak_intervals),
    }


def extract_target_and_features(audio_dict, sr):

    features_list = []

    for file_name, signal in audio_dict.items():
        feats = extract_features(signal, sr)

        df = pd.DataFrame(feats, index=[file_name])

        features_list.append(df)

    features_dataframe = pd.concat(features_list, axis=0)

    features_dataframe["anomaly"] = np.where(
        features_dataframe.index.str.contains("anomaly"),
        1,
        0
    )

    return features_dataframe


def detection_error(predictions, real_labels):
    counts = pd.DataFrame(np.where(predictions == real_labels, 1, 0), columns=["hits"]).value_counts()
    return round((counts[0]/(counts[1] + counts[0])), 3)


def compute_metrics(model_name, y_true, y_pred, scores):

    return {
        "model": model_name,
        "auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def bandpass_filter(y, low, high, sr):
    """
    Bandpass Butterworth filter.

    Interpretação: seleciona faixa espectral relevante do sinal.
    """
    nyq = 0.5 * sr
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, y)


def lowpass_filter(y, cutoff, sr):
    """
    Low-pass Butterworth filter.

    Interpretação: remove portadora e preserva modulação lenta (envelope dynamics).
    """
    nyq = 0.5 * sr
    b, a = butter(2, cutoff / nyq, btype='low')
    return filtfilt(b, a, y)


def envelope_demodulation_pipeline(y, sr, low, high, cutoff=200):
    """
    Envelope-based signal conditioning pipeline.

    Pipeline:
        1. Bandpass filtering (isola banda de interesse)
        2. Hilbert transform (construção de sinal analítico)
        3. Envelope extraction (magnitude)
        4. Low-pass filtering (extração de modulação)

    Interpretação física:
        extrai dinâmica de falhas mecânicas via análise de envelope.
    """

    # 1) bandpass filtering
    bandpassed = bandpass_filter(y, low, high, sr)

    # 2) analytic signal (Hilbert transform)
    analytic_signal = hilbert(bandpassed)

    # 3) envelope extraction
    envelope = np.abs(analytic_signal)

    # 4) smoothing (modulation extraction)
    smooth_envelope = lowpass_filter(envelope, cutoff, sr)

    return smooth_envelope


def apply_envelope_pipeline_to_audio_dict(audio_dict, low, high, cutoff=200, sr=16000):
    """
    Applies envelope demodulation pipeline to a dictionary of audio signals.
    Preserves original keys.
    """
    processed_dict = {}

    for key, y in audio_dict.items():
        processed_dict[key] = envelope_demodulation_pipeline(y, sr=sr, low=low, cutoff= cutoff, high=high)

    return processed_dict


def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_ecdf_anomaly_detection(
    test_errors,
    y_test,
    threshold=None,
    save_path=None,
    dpi=300,
    title="ECDF of Reconstruction Error"
):
    normal = test_errors[y_test.values.flatten() == 0]
    anomaly = test_errors[y_test.values.flatten() == 1]

    x1, y1 = ecdf(normal)
    x2, y2 = ecdf(anomaly)

    plt.figure(figsize=(6, 4))

    plt.plot(x1, y1, label="Normal")
    plt.plot(x2, y2, label="Anomaly")

    if threshold is not None:
        plt.axvline(threshold, color='r', linestyle='--',
                    label=f'Threshold ({threshold:.2f})')

    plt.title(title)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()


def plot_pca_feature_importance(
    pca,
    x_train,
    save_path=None,
    dpi=300,
    title="PCA Feature Importance (Mean Absolute Loadings)"
):
    # Loadings matrix
    loadings = pd.DataFrame(
        pca.components_.T,
        index=x_train.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    # Importance score
    feature_importance = np.abs(loadings).mean(axis=1).sort_values(ascending=True)

    # Plot
    plt.figure(figsize=(8, 4))
    feature_importance.plot(kind='barh')

    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()

    # Save figure if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.close()

    return feature_importance


def evaluate_and_save_metrics(
    test_errors,
    y_test,
    threshold=1.5,
    save_path=None,
    model_name="pca_reconstruction"
):

    test_predictions = (test_errors > threshold).astype(int)
    test_metrics = compute_metrics(
        model_name,
        y_test.values,
        test_predictions,
        test_errors)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(test_metrics, dict):
            test_metrics = pd.DataFrame([test_metrics])
        test_metrics.to_csv(save_path, index=False)

    return test_metrics


def plot_envelope_analysis(groups, sr, cutoff, save_path=None, dpi=300):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=dpi)
    max_f = cutoff * 2

    for i, (name, data) in enumerate(groups.items()):
        if not data: continue
        
        # Time Domain
        orig, env = data[0]
        t = np.linspace(0, len(orig)/sr, len(orig))
        
        ax_t = axes[i, 0]
        ax_t.plot(t, orig, label="Original Signal", alpha=0.3, color='gray')
        ax_t.plot(t, env, label=f"Envelope ({cutoff}Hz Cutoff)", color='darkorange', lw=1.5)
        ax_t.set_title(f"Time Domain - {name.upper()} Noise")
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("Amplitude")
        ax_t.legend(loc='upper right')
        ax_t.grid(alpha=0.2)

        # Frequency Domain
        ax_f = axes[i, 1]
        mags = []
        for _, e in data:
            f, m = rfftfreq(len(e), 1/sr)[1:], np.abs(rfft(e))[1:] / len(e)
            mags.append(m)
        
        mean_m = np.mean(mags, axis=0)
        ax_f.plot(f, mean_m, color='darkorange', lw=1.5)
        ax_f.fill_between(f, 0, mean_m, alpha=0.2, color='darkorange')
        
        # Optional: ax_f.set_yscale('log') # Uncomment for Log Magnitude
        ax_f.set_xlim(0, max_f)
        ax_f.set_title(f"Modulation Spectrum - {name.upper()} Noise")
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("Magnitude")
        ax_f.grid(alpha=0.2)

    plt.tight_layout()
    
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=dpi)
        print(f"Plot saved to: {path}")
    
    plt.show()
