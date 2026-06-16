import re
import pandas as pd

class DataTransformer:
    def __init__(self, data):
        self.data = data
        self.processor_mapping = {
            'AMD Ryzen 3': 1, 'Intel i3': 2, 'Intel i5': 3, 'AMD Ryzen 5': 4,
            'AMD Ryzen 7': 5, 'Intel i7': 6, 'AMD Ryzen 9': 7, 'Intel i9': 8
        }
        self.inverse_processor_mapping = {v: k for k, v in self.processor_mapping.items()}

        self.gpu_mapping = {
            "Integrated": 1, "Nvidia GTX 1650": 2, "Nvidia RTX 2060": 3,
            "AMD Radeon RX 6600": 4, "Nvidia RTX 3060": 5,
            "AMD Radeon RX 6800": 6, "Nvidia RTX 3080": 7
        }
        self.inverse_gpu_mapping = {v: k for k, v in self.gpu_mapping.items()}

    def transform(self):
        """Encodes the 'Processor', 'Storage', 'GPU', and 'Resolution' columns in the given DataFrame."""
        if 'Processor' in self.data.columns:
            self.data['Processor'] = self.data['Processor'].map(self.processor_mapping)

        if 'Storage' in self.data.columns:
            self.data[['Storage_Size', 'SSD', 'HDD']] = pd.DataFrame(
                self.convert_storage_to_ml_format(self.data['Storage'].tolist()), index=self.data.index
            )
            self.data.drop(columns=['Storage'], inplace=True)

        if 'GPU' in self.data.columns:
            self.data['GPU'] = self.data['GPU'].map(self.gpu_mapping)

        if 'Resolution' in self.data.columns:
            self.data['Resolution'] = self.data['Resolution'].apply(self.convert_resolution)

        return self.data

    def inverse_transform(self):
        """Decodes numerical values back to their original labels for 'Processor' and 'GPU', and reconstructs 'Storage'."""
        if 'Processor' in self.data.columns:
            self.data['Processor'] = self.data['Processor'].map(self.inverse_processor_mapping)

        if 'GPU' in self.data.columns:
            self.data['GPU'] = self.data['GPU'].map(self.inverse_gpu_mapping)

        if {'Storage_Size', 'SSD', 'HDD'}.issubset(self.data.columns):
            self.data['Storage'] = self.data.apply(lambda row: self.reconstruct_storage(row), axis=1)
            self.data.drop(columns=['Storage_Size', 'SSD', 'HDD'], inplace=True)

        return self.data

    def convert_storage_to_ml_format(self, storage_list):
        """Converts storage descriptions to a machine learning-compatible format."""
        data = []

        for item in storage_list:
            match = re.match(r"(\d+)(TB|GB)\s+(SSD|HDD)", item)
            if match:
                size, unit, storage_type = match.groups()
                size = int(size)

                if unit == "TB":
                    size *= 1024

                ssd = 1 if storage_type == "SSD" else 0
                hdd = 1 if storage_type == "HDD" else 0

                data.append([size, ssd, hdd])

        return data

    def reconstruct_storage(self, row):
        """Reconstructs the storage description from numerical format."""
        size = row['Storage_Size']
        storage_type = 'SSD' if row['SSD'] == 1 else 'HDD'
        unit = 'TB' if size >= 1024 else 'GB'
        size = size // 1024 if unit == 'TB' else size
        return f"{size}{unit} {storage_type}"

    def convert_resolution(self, resolution):
        """Converts resolution from string format '1920x1080' to integer (total pixels)."""
        a, b = map(int, resolution.split('x'))
        return a * b