import os


class DataSet:
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.data_mapping = {}
        self.label_to_idx = {}
        self.idx_to_label = {}

        self.valid_classes = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "silence",
            "unknown",
        ]
        self.label_to_idx = {label: i for i, label in enumerate(self.valid_classes)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

    def map_data(self):
        for root, dirs, files in os.walk(self.main_folder):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    class_name = os.path.basename(root)

                    if class_name not in self.valid_classes:
                        class_name = "unknown"

                    label_idx = self.label_to_idx[class_name]
                    self.data_mapping[file_path] = label_idx
