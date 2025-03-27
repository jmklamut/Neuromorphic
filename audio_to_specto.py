import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\jmkla\Downloads\Sound\Dataset_v3_sound"
output_path = r"C:\Users\jmkla\Documents\GradSchool\Neuromorphic\FinalProject\Neuromorphic\Mel_Spectrograms"

# Audio processing parameters
SAMPLE_RATE = 44100  # Standard sample rate
DURATION_MS = 4000  # Fixed audio length in ms
N_MELS = 64  # Number of Mel bands
N_FFT = 1024  # FFT window size
HOP_LENGTH = None  # Default hop length
SHIFT_LIMIT = 0.4  # Time shift factor (20%)

# Audio Utility Class
class AudioUtil:
    @staticmethod
    def open(audio_file):
        """Load an audio file."""
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel=2):
        """Convert mono to stereo or vice versa."""
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])
        return (resig, sr)

    @staticmethod
    def resample(aud, new_sr=SAMPLE_RATE):
        """Resample audio to a fixed sample rate."""
        sig, sr = aud
        if sr == new_sr:
            return aud
        resig = T.Resample(sr, new_sr)(sig)
        return (resig, new_sr)

    @staticmethod
    def pad_trunc(aud, max_ms=DURATION_MS):
        """Pad or truncate audio to a fixed duration."""
        sig, sr = aud
        max_len = sr // 1000 * max_ms
        sig_len = sig.shape[1]

        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_len = max_len - sig_len
            pad_begin = torch.zeros((sig.shape[0], random.randint(0, pad_len)))
            pad_end = torch.zeros((sig.shape[0], pad_len - pad_begin.shape[1]))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    @staticmethod
    def time_shift(aud, shift_limit=SHIFT_LIMIT):
        """Apply time shift augmentation."""
        sig, sr = aud
        shift_amt = int(random.random() * shift_limit * sig.shape[1])
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(aud, n_mels=N_MELS, n_fft=N_FFT, hop_len=HOP_LENGTH):
        """Generate Mel Spectrogram."""
        sig, sr = aud
        spec = T.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = T.AmplitudeToDB(top_db=80)(spec)  # Convert to decibel scale
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        """Apply frequency & time masking (SpecAugment)."""
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec.clone()

        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            aug_spec = T.FrequencyMasking(freq_mask_param)(aug_spec)

        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            aug_spec = T.TimeMasking(time_mask_param)(aug_spec)

        return aug_spec

# Process all audio files
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    output_category_path = os.path.join(output_path, category)

    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                
                file_path = os.path.join(category_path, file)

                if not os.path.exists(file_path):
                    print("Error: File does not exist!")
                else:
                    print("File found!")
                    
                metadata = torchaudio.info(file_path)
                
                # Load and preprocess audio
                aud = AudioUtil.open(file_path)
                aud = AudioUtil.rechannel(aud)
                aud = AudioUtil.resample(aud)
                aud = AudioUtil.pad_trunc(aud)
                aud = AudioUtil.time_shift(aud)

                # Convert to mel spectrogram
                mel_spec = AudioUtil.spectro_gram(aud)
                mel_spec = AudioUtil.spectro_augment(mel_spec)
                mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)  # Convert to dB scale

                # Save as image
                plt.figure(figsize=(6, 4))
                plt.imshow(mel_spec_db[0].numpy(), cmap='viridis', aspect='auto')
                plt.axis('off')  # Hide axes
                output_file = os.path.join(output_category_path, file.replace(".wav", ".png"))
                print(output_file)
                plt.savefig(output_file)
                plt.close()

print("Mel spectrogram conversion complete!")
