import json
import os

class State:
    def __init__(self):
        self.image_tags = None
        self.bboxes = None
        self.mask = None
        self.image_folder_path = ""
        self.label_folder_path = ""
        self.class_names = []
        self.colors = []
        self.image_files = []
        self.label_files = []
        self.task_type = ""
        self.image_index = 0
        self.image_files_index = []
        self.image_tags = {}
        self.default_index = 0
        self.state_file = 'state.json'
        self.load_state()
        self.edit_label = None
        self.new_bbox = None
        self.new_class_name = None
        self.edit_bbox_index = None

    def reset_state(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.image_index = state.get('image_index')
                self.image_tags = state.get('image_tags')
        else:
            self.image_index = 0
            self.image_tags = {}

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({'image_index': self.image_index, 'image_tags': self.image_tags}, f, indent=4)

    def edit_labels(self):
        if self.edit_label:
            if self.new_class_name:
                self.bboxes[self.edit_bbox_index]['class_name'] = self.new_class_name

            if self.new_bbox:
                xmin, ymin, xmax, ymax = [int(coord.strip()) for coord in self.new_bbox.split(",")]
                self.bboxes[self.edit_bbox_index]['bbox'] = (xmin, ymin, xmax, ymax)


    def convert_and_export_labels(self, src_format, dst_format):
        pass