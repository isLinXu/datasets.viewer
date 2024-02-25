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
        if state.task_type == "分类":
            visual_classification(state, st)
            load_image_preview(state, st)
        elif state.task_type == "检测":
            visual_detection(state, st)
            load_image_preview(state, st)
        elif state.task_type == "分割":
            visual_segmentation(state, st)
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