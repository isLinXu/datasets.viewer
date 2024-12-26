import sys
sys.path.insert(0, '/utils/')
from utils.layers import *
from utils.loads import *
from utils.visual import *
import streamlit as st
from utils.states import State
class DatasetViewer:
    def __init__(self):
        self.state = State()
        self.st = st

    def loads(self):
        load_sidebar(self.state, self.st)

    def visual(self):
        state = self.state
        st = self.st
        task_visualization = {
            "分类": visual_classification,
            "检测": visual_detection,
            "分割": visual_segmentation,
            "增强": visual_data_aug
        }
        visual_function = task_visualization.get(state.task_type)
        if visual_function:
            visual_function(state, st)
            load_image_preview(state, st)
        else:
            st.warning("请选择任务类型")
def main():
    st.title("Dataset.Viewer©️(Hertz)")
    viewer = DatasetViewer()
    viewer.loads()
    viewer.visual()

if __name__ == "__main__":
    main()