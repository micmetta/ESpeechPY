import re
import parselmouth
import pandas as pd
import numpy as np
from scipy import stats
from parselmouth.praat import call
from collections import OrderedDict


class PraatFeatures:
    """
    Extract all Praat features using Praat-Parselmouth package
    """
    # Initializer / Instance Attributes
    def __init__(self):
        self.sound = None

    @staticmethod
    def duration_energy_power(sound):
        """
        :return: basic prosodic sound features
        """
        sound = parselmouth.Sound(sound)
        prosodic_dict = OrderedDict()
        prosodic_dict['duration'] = sound.duration
        prosodic_dict['energy'] = sound.get_energy()
        prosodic_dict['power'] = sound.get_power()
        return prosodic_dict

    @staticmethod
    def pitch_features(sound):
        """
        Basic pitch features

        Returns
        ---------

        Dictionary object of pitch features
        """
        sound = parselmouth.Sound(sound)
        pitch = sound.to_pitch()
        pitch_clean = pitch.selected_array['frequency']
        pitch_mean, pitch_max, pitch_std, pitch_range = np.mean(pitch_clean), \
                                                        np.max(pitch_clean), \
                                                        np.std(pitch_clean), \
                                                        np.max(pitch_clean) - np.min(pitch_clean)
        pitch_quant = pd.Series(pitch_clean).quantile(0.70)
        pitch_diff_max_min = np.max(pitch_clean) - np.min(pitch_clean)
        pitch_diff_max_mean = np.max(pitch_clean) - np.mean(pitch_clean)
        pitch_diff_max_mode = np.max(pitch_clean) - stats.mode(pitch_clean).mode[0]

        pitch_dict = OrderedDict()
        pitch_feature = {
            "pitch_max": pitch_max,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_quant": pitch_quant,
            "pitch_diff_max_min": pitch_diff_max_min,
            "pitch_diff_max_mean": pitch_diff_max_mean,
            "pitch_diff_max_mode": pitch_diff_max_mode}
        pitch_dict.update(pitch_feature)
        return pitch_dict

    @staticmethod
    def intensity_features(sound):
        """
        Basic intensity features

        Returns
        ---------

        Dictionary object of intensity features
        """
        sound = parselmouth.Sound(sound)
        intensity = sound.to_intensity()
        intensity_clean = intensity.values.flatten()
        intensity_min = np.min(intensity_clean)
        intensity_mean, intensity_max, intensity_std = np.mean(intensity_clean), np.max(intensity_clean), np.std(
            intensity_clean)
        intensity_diff_max_min = np.max(intensity_clean) - np.min(intensity_clean)
        intensity_diff_max_mean = np.max(intensity_clean) - np.mean(intensity_clean)
        intensity_diff_max_mode = np.max(intensity_clean) - stats.mode(intensity_clean).mode[0]
        intensity_quant = pd.Series(intensity_clean).quantile(0.50)
        intensity_dict = OrderedDict()
        intensity_features = {
            "intensity_min": intensity_min,
            "intensity_mean": intensity_mean,
            "intensity_max": intensity_max,
            "intensity_std": intensity_std,
            "intensity_quant": intensity_quant,
            "intensity_diff_max_min": intensity_diff_max_min,
            "intensity_diff_max_mean": intensity_diff_max_mean,
            "intensity_diff_max_mode": intensity_diff_max_mode}
        intensity_dict.update(intensity_features)
        return intensity_dict

    @staticmethod
    def frequency_band(sound):
        """
        Basic frequency features

        Returns
        ---------

        Dictionary object of frequency features
        """
        sound = parselmouth.Sound(sound)
        formant = call(sound, "To Formant (burg)", 0.0, 5.0, 5500, 0.025, 50)
        down_to_table = call(formant, "Down to Table", 6, True, 6, False, 3, True, 3, True)
        list_formant = call(formant, "List", 6, True, 6, False, 3, True, 3, True)
        main_collection = []
        for line in list_formant.split('\n'):
            main_collection.append(line.split('\t'))
        #     print("*****************************************************")
        table = main_collection
        headers = table.pop(0)
        collection_table = pd.DataFrame(table, columns=headers)
        collection_table = collection_table.replace('--undefined--', 0)
        for col in collection_table.columns:
            collection_table[col] = pd.to_numeric(collection_table[col])

        collection_table_main = collection_table.describe()
        frequency_dict = OrderedDict()
        freq_data = {
            "avgVal1": collection_table_main.loc['mean', 'F1(Hz)'],
            "avgVal2": collection_table_main.loc['mean', 'F2(Hz)'],
            "avgVal3": collection_table_main.loc['mean', 'F3(Hz)'],
            "avgBand1": collection_table_main.loc['mean', 'B1(Hz)'],
            "avgBand2": collection_table_main.loc['mean', 'B2(Hz)'],
            "avgBand3": collection_table_main.loc['mean', 'B3(Hz)'],
            "fmean1": collection_table_main.loc['mean', 'F1(Hz)'],
            "fmean2": collection_table_main.loc['mean', 'F2(Hz)'],
            "fmean3": collection_table_main.loc['mean', 'F3(Hz)'],
            "f2meanf1": collection_table_main.loc['mean', 'F2(Hz)'] / (collection_table_main.loc['mean', 'F1(Hz)'] * 1.0),
            "f3meanf1": collection_table_main.loc['mean', 'F3(Hz)'] / (collection_table_main.loc['mean', 'F1(Hz)'] * 1.0),
            "f1STD": collection_table_main.loc['std', 'F1(Hz)'],
            "f2STD": collection_table_main.loc['std', 'F2(Hz)'],
            "f3STD": collection_table_main.loc['std', 'F3(Hz)'],
            "f2STDf1": collection_table_main.loc['std', 'F2(Hz)'] / (collection_table_main.loc['std', 'F1(Hz)'] * 1.0),
            "f3STDf1": collection_table_main.loc['std', 'F3(Hz)'] / (collection_table_main.loc['std', 'F1(Hz)'] * 1.0)

                 }
        frequency_dict.update(freq_data)
        return frequency_dict

    @staticmethod
    def jitter_shimmer(sound):
        """
        Basic jitter and shimmer features

        Returns
        ---------

        Dictionary object of jitter and shimmer features
        """
        sound = parselmouth.Sound(sound)
        pitch_cc = call(sound, "To Pitch (cc)", 0.0, 75, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, 600)
        pitch_point = call([sound, pitch_cc], "To PointProcess (cc)")
        jitter_shimmer_output = call([sound, pitch_cc, pitch_point], "Voice report", 0.0, 0.0, 75.0, 600, 1.3, 1.6, 0.03, 0.45)
        jitter_shimmer_output_str = str(jitter_shimmer_output)
        jns_list = ['Jitter local:', 'Shimmer local:', 'Jitter rap:',
                    'Mean period:', 'Number of voice breaks:', 'Fraction of locally unvoiced frames:',
                    'Degree of voice breaks:']
        jitter_shimmer_dict = OrderedDict()
        for word in jns_list:
            jitter_shimmer_output_cleaned = jitter_shimmer_output_str.replace('(', '').replace(')', '')
            pattern = word + ' ([E0-9-.%]+)'
            match = re.search(pattern, jitter_shimmer_output_cleaned)
            match_ans = match.group(1)
            if '%' in match_ans:
                match_ans = float(match_ans.strip('%')) / 100.0
            elif 'E' in match_ans:
                match_ans = float(match_ans)
            word = word.replace(':', '').replace(' ', '_')
            jitter_shimmer_dict[word] = match_ans
        return jitter_shimmer_dict

    def get_all_sound_feature(self, sound):
        """
        :return: Output dictionary of all features
        """
        sound = parselmouth.Sound(sound)
        sound_dictionary = dict()
        sound_dictionary['duration_energy_power'] = self.duration_energy_power(sound)
        sound_dictionary['pitch_features'] = self.pitch_features(sound)
        sound_dictionary['intensity_features'] = self.intensity_features(sound)
        sound_dictionary['frequency_features'] = self.frequency_band(sound)
        sound_dictionary['jitter_shimmer'] = self.jitter_shimmer(sound)
        return sound_dictionary
