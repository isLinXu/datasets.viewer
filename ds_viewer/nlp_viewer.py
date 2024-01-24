import streamlit as st
from datasets import load_dataset
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 加载问答数据集
@st.cache_data
def load_qa_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

# 加载多模态数据集
@st.cache_data
def load_mm_dataset(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

# 生成词云
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    return wordcloud

def display_wordcloud(wordcloud):
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# 主界面
def main():
    st.title("Huggingface Dataset Viewer")
    st.sidebar.title("Options")

    # 对话的图标
    user_avatar = "🧑‍💻"
    robot_avatar = "🤖"

    # 初始化消息列表和当前索引
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # 选择数据集类型
    dataset_type = st.sidebar.selectbox("Select Dataset Type", ["Question-Answering", "Multimodal"])

    # 选择数据集分割
    split = st.sidebar.selectbox("Select Dataset Split", ["train", "validation"])

    # 根据选择的数据集类型加载数据
    if dataset_type == "Question-Answering":
        dataset_name = st.sidebar.text_input("Enter QA Dataset Name", "squad")
        dataset = load_qa_dataset(dataset_name, split)
    else:
        dataset_name = st.sidebar.text_input("Enter Multimodal Dataset Name", "coco")
        dataset = load_mm_dataset(dataset_name, split)

    # 显示数据集大小
    st.write(f"Dataset Size: {len(dataset)}")

    # 控制数据集索引
    index = st.sidebar.number_input("Index", min_value=0, max_value=len(dataset)-1, step=1, value=st.session_state.current_index)

    # 如果索引改变，更新对话记录
    if st.session_state.current_index != index:
        st.session_state.current_index = index
        st.session_state.messages = []

    # 显示选择的数据
    data = dataset[index]
    if dataset_type == "Question-Answering":
        st.text_area("Context", value=data['context'], height=200)
        st.write("---")
        if not st.session_state.messages:
            st.session_state.messages.append({"role": "user", "content": data['question'], "avatar": user_avatar})
            st.session_state.messages.append({"role": "robot", "content": data['answers']['text'][0], "avatar": robot_avatar})

        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])

    else:
        st.image(data['image'])
        st.write(f"Caption: {data['caption']}")

    # 是否生成并显示词云
    show_wordcloud = st.sidebar.checkbox("Show Word Cloud")
    if show_wordcloud and dataset_type == "Question-Answering":
        text = " ".join([data['context'] for data in dataset])
        wordcloud = generate_wordcloud(text)
        display_wordcloud(wordcloud)

if __name__ == "__main__":
    main()