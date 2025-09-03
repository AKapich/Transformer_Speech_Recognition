import os
import random


class DataSet:
    def __init__(self, main_folder, seed=42):
        self.main_folder = main_folder
        self.data_mapping = {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.seed = seed

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

    def split_random(self, proportions=(0.8, 0.1, 0.1)):
        """Random 80/10/10 split"""
        if self.data_mapping == {}:
            self.data_mapping = self.map_data()
        items = list(self.data_mapping.items())
        random.seed(self.seed)
        random.shuffle(items)

        n = len(items)
        n_train = int(n * proportions[0])
        n_val = int(n * proportions[1])

        train = dict(items[:n_train])
        val = dict(items[n_train : n_train + n_val])
        test = dict(items[n_train + n_val :])

        return train, val, test

    def split_with_lists(self, validation_list, testing_list):
        """
        Split using predefined txt files (paths relative to main_folder).
        Silence files are divided randomly into 80/10/10.
        """
        if self.data_mapping == {}:
            self.data_mapping = self.map_data()

        with open(validation_list) as f:
            val_set = set(line.strip() for line in f)
        with open(testing_list) as f:
            test_set = set(line.strip() for line in f)

        train, val, test = {}, {}, {}

        for file_path, label in self.data_mapping.items():
            rel_path = os.path.relpath(file_path, self.main_folder).replace("\\", "/")
            class_name = self.idx_to_label[label]

            if class_name == "silence":
                continue

            if rel_path in val_set:
                val[file_path] = label
            elif rel_path in test_set:
                test[file_path] = label
            else:
                train[file_path] = label

        # Handle silence separately
        silence_files = [
            fp
            for fp, lbl in self.data_mapping.items()
            if self.idx_to_label[lbl] == "silence"
        ]
        random.seed(self.seed)
        random.shuffle(silence_files)

        n = len(silence_files)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        for fp in silence_files[:n_train]:
            train[fp] = self.label_to_idx["silence"]
        for fp in silence_files[n_train : n_train + n_val]:
            val[fp] = self.label_to_idx["silence"]
        for fp in silence_files[n_train + n_val :]:
            test[fp] = self.label_to_idx["silence"]

        return train, val, test
